"""
rgbir_dataset.py

Paired RGB + LWIR loaders for LLVIP and KAIST, used for self-supervised
pretraining only (encoder_only_mode=true).

Both datasets are pixel-registered (LLVIP by homography, KAIST by beam
splitter), so the RGB and thermal crops share the same normalized crop box
without the FLIR-specific alignment offsets. Views, flips and augmentations are
inherited from FlirAdasDataset so the RGB-thermal recipe matches the paper:
synchronized crops/flip for both streams, photometric augmentations on RGB
only, 1-channel normalized thermal.

Split selection mirrors the MJEPA loaders (src/datasets/{llvip,kaist}.py):
- LLVIP: visible/{train,test} + infrared/{train,test}, same file names.
- KAIST: imageSets/<split>.txt listing set/video/frame ids, images under
  images/setXX/VXXX/{visible,lwir}/IXXXXX.jpg.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

from src.flir_dataset import FlirAdasDataset, flir_collate_fn

rgbir_collate_fn = flir_collate_fn


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v) for v in value]


def list_llvip_pairs(root: str, modes: Sequence[str]) -> List[Tuple[Path, Path]]:
    root = Path(root)
    pairs = []
    for mode in modes:
        rgb_dir = root / "visible" / mode
        ir_dir = root / "infrared" / mode
        if not rgb_dir.is_dir():
            print(f"⚠️  LLVIP: {rgb_dir} not found, skipping split '{mode}'")
            continue
        kept = 0
        for name in sorted(os.listdir(rgb_dir)):
            rgb_path = rgb_dir / name
            ir_path = ir_dir / name
            if ir_path.exists():
                pairs.append((rgb_path, ir_path))
                kept += 1
        print(f"LLVIP split '{mode}': {kept} pairs")
    return pairs


def list_kaist_pairs(root: str, split_files: Sequence[str]) -> List[Tuple[Path, Path]]:
    root = Path(root)
    pairs = []
    for split_file in split_files:
        split_path = root / "imageSets" / split_file
        if not split_path.exists():
            print(f"⚠️  KAIST: {split_path} not found, skipping")
            continue
        with open(split_path, "r") as f:
            img_ids = [line.strip() for line in f if line.strip()]
        kept = 0
        for img_id in img_ids:
            set_value, vid_value, img_name = img_id.split("/")[:3]
            base = root / "images" / set_value / vid_value
            rgb_path = base / "visible" / f"{img_name}.jpg"
            ir_path = base / "lwir" / f"{img_name}.jpg"
            if not rgb_path.exists() or not ir_path.exists():
                raise FileNotFoundError(f"KAIST pair missing for id '{img_id}' under {base}")
            pairs.append((rgb_path, ir_path))
            kept += 1
        print(f"KAIST split '{split_file}': {kept} pairs")
    return pairs


class PairedRGBIRDataset(FlirAdasDataset):
    """Registered RGB/LWIR pairs with the Le MuMo multi-crop view contract."""

    def __init__(
        self,
        pairs: Iterable[Tuple[Path, Path]],
        name: str,
        split: str = "train",
        arch: str = "C",
        V: int = 2,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 4,
        img_size: int = 224,
        local_img_size: int = 96,
        probe_img_size: int | None = None,
        modality_dropout: float = 0.0,
        include_probe_view: bool = False,
        dino_aug_mode: str = "default",
    ):
        # FlirAdasDataset.__init__ reads FLIR metadata, so only reuse its
        # transform/view helpers and set the attributes they rely on.
        self.pairs = [{"rgb_path": Path(r), "thermal_path": Path(t)} for r, t in pairs]
        if not self.pairs:
            raise RuntimeError(f"No {name} RGB/IR pairs found; check the dataroot and split names.")
        self.split = str(split).lower()
        self.arch = arch
        self.V = V
        self.global_crops_scale = tuple(global_crops_scale)
        self.local_crops_scale = tuple(local_crops_scale)
        self.local_crops_number = local_crops_number
        self.img_size = img_size
        self.local_img_size = local_img_size
        self.probe_img_size = int(probe_img_size or img_size)
        self.modality_dropout = modality_dropout
        self.finetune_mode = False
        self.include_probe_view = bool(include_probe_view)
        self.dino_aug_mode = str(dino_aug_mode).lower()
        self.use_official_dino_augs = self.split == "train" and self.dino_aug_mode == "official"
        self.resize_mode = "center_crop"
        self.align_modalities = False  # already pixel-registered
        self.lidar_mode = "depth"
        self.num_scenes = 1
        self.num_cameras = 1
        self.num_locations = 1
        self._setup_transforms()
        print(
            f"PairedRGBIRDataset [{name}, arch={arch}, modality=thermal]: {len(self.pairs)} paired samples, "
            f"Global={V}, Local={local_crops_number}"
        )

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        rgb_img = Image.open(pair["rgb_path"]).convert("RGB")
        thermal_img = Image.open(pair["thermal_path"]).convert("L")

        rgb_global, rgb_local, thermal_global, thermal_local, rgb_probe, thermal_probe = self._make_synced_views(
            rgb_img, thermal_img
        )
        cam_views = {"global": rgb_global, "local": rgb_local, "probe": rgb_probe}
        modality2 = {"global": thermal_global, "local": thermal_local, "probe": thermal_probe}

        if self.modality_dropout > 0 and self.split == "train" and np.random.random() < self.modality_dropout:
            modality2 = {k: v.new_zeros(v.shape) for k, v in modality2.items()}

        return cam_views, modality2, self._get_global_labels(pair)


DEFAULT_SPLITS = {
    "llvip": ["train", "test"],
    "kaist": ["train-all-04.txt", "test-all-04.txt"],
}


def list_rgbir_pairs(name: str, root: str, splits=None) -> List[Tuple[Path, Path]]:
    name = str(name).lower()
    splits = _as_list(splits) or DEFAULT_SPLITS.get(name, [])
    if name == "llvip":
        return list_llvip_pairs(root, splits)
    if name == "kaist":
        return list_kaist_pairs(root, splits)
    raise ValueError(f"Unknown RGB/IR dataset '{name}'. Use 'llvip' or 'kaist'.")


def build_rgbir_dataset(sources, **kwargs) -> PairedRGBIRDataset:
    """sources: list of (name, root, splits). Several sources are concatenated
    into one pair list, like ConcatDataset in the 4-JEPA rgb_ir loader."""
    pairs = []
    for name, root, splits in sources:
        pairs.extend(list_rgbir_pairs(name, root, splits))
    return PairedRGBIRDataset(pairs, name="+".join(str(n).lower() for n, _, _ in sources), **kwargs)
