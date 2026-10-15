"""Sequence datasets for temporal driving policies.

The raw meta files only store {"img", "action"} -- there is no explicit frame
index in the JSON. However the filenames encode episode and step
(`run01_000042.png`), so consecutive samples can be recovered by parsing them.
Windows must never straddle an episode boundary, so samples are grouped per run
and windows are taken inside each run.

Provides:
    SequenceDataset      -> (T, 3, H, W) windows for frame-stacked / GRU models
    make_windows         -> the window index construction (shared logic)
"""
from __future__ import annotations

import json
import os
import re
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

_FNAME_RE = re.compile(r"run(\d+)_(\d+)\.png$")


class SequenceDataset(Dataset):
    """Yields temporal windows of frames plus the action at the LAST frame.

    Each item: (frames (T,3,H,W) float32 normalized, action (3,))
    A window is only formed from consecutive frames of the same episode.
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, raw_dir: str, img_size: Tuple[int, int] = (80, 160),
                 n_frames: int = 4, stride: int = 1, dagger_dir: str | None = None,
                 preload: bool = True):
        self.raw_dir = raw_dir
        self.img_size = img_size
        self.n_frames = n_frames
        self.preload = preload
        self._cache: dict[str, np.ndarray] = {}

        episodes = self._scan_episodes(raw_dir, dagger_dir)
        # build windows inside each episode
        self.windows: List[Tuple[List[str], np.ndarray]] = []
        for ep_files, ep_actions in episodes:
            n = len(ep_files)
            for end in range(n_frames - 1, n, stride):
                idxs = list(range(end - n_frames + 1, end + 1))
                paths = [ep_files[i] for i in idxs]
                act = ep_actions[idxs[-1]]
                self.windows.append((paths, act))

    # ---------------------------------------------------------------- scanning
    @staticmethod
    def _scan_episodes(raw_dir, dagger_dir):
        """Return [(list_of_frame_paths_sorted_by_step, actions), ...] per run."""
        metas = sorted(glob(os.path.join(raw_dir, "meta_run*.jsonl")))
        if dagger_dir:
            metas += sorted(glob(os.path.join(dagger_dir, "meta_dagger_*.jsonl")))
        per_ep: dict[str, List[Tuple[int, str, np.ndarray]]] = {}
        for meta_path in metas:
            root = os.path.dirname(meta_path)
            with open(meta_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    p = os.path.join(root, rec["img"])
                    if not os.path.exists(p):
                        continue
                    m = _FNAME_RE.search(rec["img"])
                    if m:
                        key, step = m.group(1), int(m.group(2))
                    else:
                        # fall back to a single synthetic episode per meta file
                        key, step = os.path.basename(meta_path), len(
                            per_ep.get(os.path.basename(meta_path), []))
                    per_ep.setdefault(key, []).append(
                        (step, p, np.asarray(rec["action"], dtype=np.float32)))
        episodes = []
        for key, items in per_ep.items():
            items.sort(key=lambda t: t[0])
            episodes.append(([p for _, p, _ in items],
                             [a for _, _, a in items]))
        return episodes

    # ---------------------------------------------------------------- loading
    def _img(self, path):
        if self.preload and path in self._cache:
            return self._cache[path]
        img = Image.open(path).convert("RGB").resize(
            (self.img_size[1], self.img_size[0]))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - np.asarray(self.MEAN)) / np.asarray(self.STD)
        t = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        if self.preload:
            self._cache[path] = t
        return t

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        paths, action = self.windows[idx]
        frames = torch.stack([self._img(p) for p in paths], dim=0)  # (T,3,H,W)
        return frames, torch.from_numpy(action)
