"""GPU-resident dataset + fast training utilities.

The original BCDataset re-decodes every PNG from disk on every epoch (PIL resize
+ float32 normalize on CPU), which starves the GPU. Since our datasets are small
(a few thousand 80x160 frames = tens of MB), we can decode once and keep the
whole thing on the GPU as a single float16/float32 tensor, then train with pure
tensor slicing and no DataLoader workers at all.

Typical speedup on a GTX 1650 for 1200 frames: ~9 it/s -> several hundred it/s.
"""
from __future__ import annotations

import json
import os
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class BCDataset(Dataset):
    """Standard on-disk dataset (kept for CPU / low-VRAM fallback)."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, raw_dir: str, img_size: Tuple[int, int] = (80, 160)):
        self.raw_dir = raw_dir
        self.img_size = img_size
        self.samples: List[Tuple[str, np.ndarray]] = []
        for meta_path in sorted(glob(os.path.join(raw_dir, "meta_run*.jsonl"))):
            with open(meta_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    img_path = os.path.join(raw_dir, rec["img"])
                    action = np.asarray(rec["action"], dtype=np.float32)
                    self.samples.append((img_path, action))

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        img = Image.open(path).convert("RGB").resize(
            (self.img_size[1], self.img_size[0])
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - np.asarray(self.MEAN)) / np.asarray(self.STD)
        return torch.from_numpy(arr.transpose(2, 0, 1)).float()

    def __getitem__(self, idx):
        img_path, action = self.samples[idx]
        return self._load(img_path), torch.from_numpy(action)


def _decode_all(paths, img_size):
    """Decode all images once into a single uint8 array (N,H,W,3)."""
    h, w = img_size
    out = np.empty((len(paths), h, w, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB").resize((w, h))
        out[i] = np.asarray(img, dtype=np.uint8)
    return out


class GPUDataset:
    """Whole dataset pre-loaded into GPU memory.

    Holds images as uint8 on device (cheap, 8-bit) and normalizes on the fly in
    the training loop, so no CPU work happens per batch and no workers are
    needed. Indexing returns a*batch of already-normalized float tensors.
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, raw_dir: str, img_size: Tuple[int, int] = (80, 160),
                 device: str = "cuda", dtype=torch.float32,
                 dagger_dir: str | None = None):
        self.img_size = img_size
        self.device = torch.device(device)
        self.dtype = dtype

        paths, actions = self._scan(raw_dir, dagger_dir)
        if not paths:
            raise SystemExit(f"no samples found in {raw_dir} (and dagger_dir={dagger_dir})")

        arr = _decode_all(paths, img_size)                     # (N,H,W,3) uint8
        self.images = torch.from_numpy(arr).to(self.device)     # stays on GPU
        self.actions = torch.from_numpy(
            np.stack(actions).astype(np.float32)).to(self.device)
        self.n = len(paths)

        m = torch.tensor(self.MEAN, device=self.device).view(1, 3, 1, 1)
        s = torch.tensor(self.STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = m, s
        print(f"[gpu-data] loaded {self.n} samples -> {self.device} "
              f"({self.images.numel() / 1e6:.1f} MB uint8)")

    @staticmethod
    def _scan(raw_dir, dagger_dir):
        paths, actions = [], []
        metas = sorted(glob(os.path.join(raw_dir, "meta_run*.jsonl")))
        if dagger_dir:
            metas += sorted(glob(os.path.join(dagger_dir, "meta_dagger_*.jsonl")))
        for meta_path in metas:
            root = os.path.dirname(meta_path)
            with open(meta_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    p = os.path.join(root, rec["img"])
                    if os.path.exists(p):
                        paths.append(p)
                        actions.append(np.asarray(rec["action"], dtype=np.float32))
        return paths, actions

    def __len__(self):
        return self.n

    def normalize(self, batch_u8):
        """uint8 (B,H,W,3) on device -> normalized float (B,3,H,W)."""
        x = batch_u8.permute(0, 3, 1, 2).to(self.dtype).div_(255.0)
        return (x - self._mean) / self._std

    def batches(self, batch_size: int, shuffle: bool = True, generator=None):
        """Yield (x, y) GPU tensors. Pure indexing, no CPU decode."""
        idx = torch.randperm(self.n, device=self.device, generator=generator) \
            if shuffle else torch.arange(self.n, device=self.device)
        for i in range(0, self.n, batch_size):
            sel = idx[i:i + batch_size]
            yield self.normalize(self.images[sel]), self.actions[sel]

    def full(self):
        """All samples as (x, y) on device (fine for small datasets)."""
        return self.normalize(self.images), self.actions
