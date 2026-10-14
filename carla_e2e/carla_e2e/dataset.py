"""PyTorch dataset for behavior-cloning from CARLA expert trajectories."""
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
    """Loads (image, action) pairs from data/raw/meta_runXX.jsonl files.

    Each jsonl line: {"img": "images/run01_000123.png",
                      "action": [steer, throttle, brake]}
    Images are normalized to ImageNet mean/std.
    """

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
