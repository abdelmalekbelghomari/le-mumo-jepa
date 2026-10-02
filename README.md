<div align="center">

<img src="assets/le-mumo-jepa-logo.png" alt="Le MuMo JEPA Logo" width="600"/>

# Le MuMo JEPA

### Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens

[![arXiv](https://img.shields.io/badge/arXiv-2603.24327-b31b1b.svg)](https://arxiv.org/abs/2603.24327)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

[**Paper**](https://arxiv.org/abs/2603.24327) | [**Website**](https://ciemcornelissen.github.io/)

</div>

---

Official code for **"Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens"**, accepted at the **CVPR 2026 Workshop on Unified Robotic Vision with Cross-Modal Sensing and Alignment (URVIS)**.

## Overview

Le MuMo JEPA is a self-supervised framework that learns unified representations from RGB images and aligned companion modalities (e.g., camera-aligned LiDAR depth, thermal). It extends [LeJEPA](https://arxiv.org/abs/2511.08544) to the multi-modal setting by introducing **learnable fusion tokens** that act as a latent bottleneck between modality-specific patch stems inside a shared Vision Transformer.

Key features:
- **Learnable fusion tokens** — a set of tokens equal in number to the spatial patches that aggregate information from spatially corresponding RGB and companion-modality patches through cross-attention.
- **Pruned fusion** (default) — after the initial cross-modal attention layer, modality-specific tokens are dropped, forcing cross-modal information into the shared fusion-token grid. This reduces attention cost by ~9x while preserving representation quality.
- **SIGReg objective** — Sketched Isotropic Gaussian Regularization applied to the joint multimodal CLS embedding, without requiring stop-gradients or teacher-student networks.

<div align="center">
<img src="assets/architecture.png" alt="Le MuMo JEPA Architecture" width="500"/>
<br>
<em>Figure: Overview of Le MuMo JEPA. The companion modality is fused with RGB through learnable fusion tokens that act as a latent bottleneck inside a shared transformer.</em>
</div>

## Qualitative Results

Le MuMo JEPA produces rich spatial representations that support multiple downstream tasks through frozen patch probes, including dense depth estimation, semantic segmentation, and CenterNet-style 3D object detection.

<div align="center">
<img src="assets/probe_bbox3d.png" alt="Probe outputs" width="800"/>
<br>
<em>Qualitative probe outputs from a frozen Le MuMo JEPA encoder: RGB input, depth prediction, panoptic segmentation, and 3D detection.</em>
</div>

## Installation

```bash
git clone https://github.com/ciemcornelissen/le-mumo-jepa.git
cd le-mumo-jepa
pip install -r requirements.txt
```

`requirements.txt` covers the core paper training path. Optional ablations such as ImageBind, Sinkhorn/GeomLoss, and THOP profiling require extra dependencies or weights.

## Usage

### Self-Supervised Pretraining

`train.py` loads [`configs/default.yaml`](configs/default.yaml). Override existing Hydra keys with `key=value`; reserve `+key=value` for new keys only.

These commands use the checked-in paper defaults for the main pruned fusion-token SIGReg setup, so only dataset roots need to be provided.

```bash
# Waymo (RGB + LiDAR depth)
python train.py \
    dataset=waymo \
    waymo_dataroot=/path/to/waymo_data

# nuScenes (RGB + LiDAR depth)
python train.py \
    dataroot=/path/to/nuscenes_data

# FLIR ADAS from scratch (RGB + Thermal, longer paper schedule)
python train.py \
    dataset=flir \
    flir_dataroot=/path/to/flir_adas_v2 \
    epochs=20

# 3-pass auxiliary SIGReg ablation
python train.py \
    dataroot=/path/to/nuscenes_data \
    fusion_skip_aux_sigreg=false
```

### LLVIP / KAIST (RGB + LWIR, SSL only)

LLVIP and KAIST have no probe labels in this pipeline, so they are used for encoder-only pretraining; the checkpoint is then evaluated on FLIR ADAS v2 with frozen probes.

```bash
# LLVIP: visible/{train,test} + infrared/{train,test}
python train.py dataset=llvip +llvip_dataroot=/path/to/LLVIP "+llvip_splits=[train,test]" \
    +encoder_only_mode=true epochs=20 +run_name=lemumo_llvip +save_root=/path/to/runs

# KAIST: imageSets/*.txt split files (default: train-all-04.txt + test-all-04.txt)
python train.py dataset=kaist +kaist_dataroot=/path/to/KAIST "+kaist_splits=[train-all-04.txt,test-all-04.txt]" \
    +encoder_only_mode=true epochs=20 +run_name=lemumo_kaist +save_root=/path/to/runs

# Frozen FLIR probing of the pretrained encoder
python train.py dataset=flir flir_dataroot=/path/to/flir_adas_v2 \
    +pretrained_encoder_path=/path/to/runs/lemumo_llvip/latest.pt +probe_only_training=true \
    V=1 local_crops_number=0 epochs=5 +probe_img_size=640
```

SLURM scripts running both steps (CRIANN / Jean Zay) are in [`slurm/`](slurm/).

### Fine-tuning

```bash
# Waymo-pretrained encoder fine-tuned on FLIR
python finetune.py \
    --checkpoint /path/to/pretrained.pth \
    --dataset flir \
    --flir_dataroot /path/to/flir_adas_v2
```

For `--dataset flir`, unset fine-tuning flags follow the paper schedule: batch size 64, 30 epochs, encoder LR `2e-5`, decoder LR `2e-4`, 3-layer 512D CenterNet, 224 train-view / 640 eval-view probe sizing, validation every 100 steps, patience 8, and end-to-end encoder updates from epoch 0. Override any of these flags explicitly if you want a different schedule.

### Configuration

Default self-supervised training configuration is in [`configs/default.yaml`](configs/default.yaml). The checked-in defaults match the paper's shared fusion-token hyperparameters and keep the main single-pass joint-CLS objective; set `fusion_skip_aux_sigreg=false` to enable the more expensive 3-pass auxiliary ablation. The FLIR from-scratch recipe still uses the paper's longer 20-epoch schedule. Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `arch` | `C` | Base architecture family used for fusion-token training |
| `fusion_tokens_sigreg` | `true` | Enables the learnable fusion-token Le MuMo JEPA variant |
| `fusion_skip_aux_sigreg` | `true` | Keeps the paper's main joint-only objective; set to `false` for the 3-pass RGB-only/LiDAR-only auxiliary ablation |
| `fusion_tokens_variant` | `prune_after_first` | Pruned fusion-token routing used in the paper |
| `aligned_mode` | `true` | Uses a 1-channel aligned companion modality input |
| `lidar_mode` | `depth` | Uses aligned depth-style inputs instead of 5-channel range images |
| `lamb` | `0.1` | SIGReg trade-off weight |
| `V` | `2` | Number of global crops |
| `local_crops_number` | `8` | Number of local crops |
| `proj_dim` | `16` | Projection dimension for SIGReg |
| `bs` | `64` | Batch size |
| `lr` | `1e-4` | Learning rate |
| `epochs` | `5` | Number of self-supervised training epochs for the shared paper setup |

## Repository Structure

```
le-mumo-jepa/
├── train.py                        # Main SSL pretraining script
├── finetune.py                     # Downstream fine-tuning / evaluation script
├── configs/
│   └── default.yaml                # Default training configuration
├── src/
│   ├── encoder.py                  # Multi-modal encoders (including MMEncoderC_FusionTokens)
│   ├── losses.py                   # Baseline losses (VICReg, InfoNCE)
│   ├── baseline_encoders.py        # Baseline encoders (DINOv3, ImageBind, MultiMAE)
│   ├── novel_regularizers.py       # Additional regularizers (GMM, Sinkhorn, Spectral)
│   ├── dataset.py                  # Multi-modal dataset (nuScenes)
│   ├── waymo_dataset.py            # Waymo dataset
│   ├── flir_dataset.py             # FLIR ADAS dataset
│   ├── lidar_utils.py              # LiDAR processing utilities
│   ├── lidar_augmentations.py      # LiDAR augmentation pipeline
│   ├── detection_probes.py         # Frozen patch probes (CenterNet, depth, segmentation)
│   ├── detection_labels.py         # Detection label processing
│   └── detection_integration.py    # Detection pipeline integration
├── assets/                         # Figures for README
├── requirements.txt
└── LICENSE
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{cornelissen2026lemumojepa,
    title     = {Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens},
    author    = {Cornelissen, Ciem and Leroux, Sam and Simoens, Pieter},
    booktitle = {CVPR 2026 Workshop on Unified Robotic Vision with Cross-Modal Sensing and Alignment (URVIS)},
    year      = {2026},
    note      = {arXiv:2603.24327}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE). If you use this code in your research, please cite our paper.

## Acknowledgments

This work received financial support from the [Flanders AI Research Program (FAIR)](https://www.flandersairesearch.be/).

**Authors:** [Ciem Cornelissen](https://ciemcornelissen.github.io/), Sam Leroux, Pieter Simoens — IDLab, Department of Information Technology, Ghent University – imec, Belgium.
