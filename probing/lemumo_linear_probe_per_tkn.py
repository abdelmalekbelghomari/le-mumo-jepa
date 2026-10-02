"""
Linear Probing de l'encodeur Le MuMo JEPA (fusion tokens, ViT-S/16) sur FLIR Aligned.
Version multi-label per-token avec mAP.

Copie de mjepa_linear_probe_per_tkn.py (MJEPA), lui-même IDENTIQUE à
jepa_linear_probe_per_tkn.py (4-JEPA) : seul le chargement de l'encodeur diffère.
Tout changement ailleurs casse la comparabilité des mAP.

Pipeline : image -> encodeur Le MuMo -> fusion tokens [B,196,384] (sortie après la
norme finale, celle lue par les patch probes du papier) -> linear per-token -> [B,196,4]

Le MuMo n'a qu'un encodeur joint : une modalité seule est obtenue en mettant
l'autre à zéro (dans l'espace normalisé), comme les évals rgb_only / lidar_only
de train.py.
  - RGB seul  : RGB + IR à zéro
  - IR seul   : RGB à zéro + IR
  - BOTH      : concat des features RGB seul + IR seul -> [B, 196, 768] (protocole 4-JEPA)
  - JOINT     : RGB + IR ensemble dans l'encodeur -> [B, 196, 384] (fusion Le MuMo,
                mode en plus, pas dans les modes par défaut)

Structure FLIR Aligned attendue :
  flir_root/
    align_train.txt
    align_validation.txt
    Annotations/
      FLIR_xxxxx_PreviewData.xml
    AnnotatedImages/
      FLIR_xxxxx_RGB.jpg
      FLIR_xxxxx_PreviewData.jpeg
"""

import datetime
import os
import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_curve, auc
import random
import sys

# src.encoder vit à la racine du repo Le MuMo
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Graine aléatoire fixée à : {seed}")


# ------------------------------------------------------------------
# W&B (logging seulement : n'influence ni les données ni l'entraînement du probe)
# ------------------------------------------------------------------

def wandb_init(args):
    """Un run W&B par seed, groupé par checkpoint. Désactivé si --wandb_project
       est vide, si wandb n'est pas installé ou si WANDB_MODE=disabled."""
    if not args.wandb_project or os.environ.get('WANDB_MODE') == 'disabled':
        return None
    try:
        import wandb
    except ImportError:
        print("⚠️  wandb non installé : logging TensorBoard uniquement")
        return None
    group = args.wandb_group or Path(args.jepa_checkpoint).resolve().parent.name
    return wandb.init(
        project=args.wandb_project,
        group=group,
        job_type='flir_aligned_probe',
        name=f"{group}_probe_seed{args.seed}",
        config=vars(args),
    )


def wandb_log(run, data, step=None):
    if run is not None:
        run.log(data, step=step)


# ------------------------------------------------------------------
# 1. PARSING XML -> LABEL MULTI-LABEL
# ------------------------------------------------------------------

FLIR_CLASSES = ['person', 'car', 'bicycle', 'background']
CLASS2IDX    = {c: i for i, c in enumerate(FLIR_CLASSES)}
IDX2CLASS    = {i: c for c, i in CLASS2IDX.items()}
NUM_CLASSES  = len(FLIR_CLASSES)


def parse_xml_to_resized_boxes(xml_path: str, target_size: int = 224):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Erreur de lecture du XML : {e}")
        return []

    orig_width = float(root.find('size/width').text)
    orig_height = float(root.find('size/height').text)

    scale_x = target_size / orig_width
    scale_y = target_size / orig_height

    boxes_list = []

    for obj in root.findall('object'):
        name_node = obj.find('name')
        if name_node is None:
            continue
            
        name = name_node.text.strip().lower()
        if name in CLASS2IDX:
            class_idx = CLASS2IDX[name]

            bndbox = obj.find('bndbox')
            xmin_orig = float(bndbox.find('xmin').text)
            ymin_orig = float(bndbox.find('ymin').text)
            xmax_orig = float(bndbox.find('xmax').text)
            ymax_orig = float(bndbox.find('ymax').text)

            xmin_scaled = xmin_orig * scale_x
            ymin_scaled = ymin_orig * scale_y
            xmax_scaled = xmax_orig * scale_x
            ymax_scaled = ymax_orig * scale_y
            boxes_list.append((class_idx, xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled))
            
    return boxes_list
 
def generate_patch_labels(xml_boxes, img_size=224, patch_size=16, num_classes=4):
    grid_dim = img_size // patch_size 
    num_patches = grid_dim * grid_dim
    
    patch_labels = np.zeros((num_patches, num_classes), dtype=np.float32)
    
    for class_idx, xmin, ymin, xmax, ymax in xml_boxes:
        c_start = int(np.clip(xmin // patch_size, 0, grid_dim - 1))
        c_end   = int(np.clip(xmax // patch_size, 0, grid_dim - 1))
        r_start = int(np.clip(ymin // patch_size, 0, grid_dim - 1))
        r_end   = int(np.clip(ymax // patch_size, 0, grid_dim - 1))
        
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                patch_idx = r * grid_dim + c
                patch_labels[patch_idx, class_idx] = 1.0

    for patch_idx in range(num_patches):
        if patch_labels[patch_idx, :num_classes-1].sum() == 0:
            patch_labels[patch_idx, num_classes-1] = 1.0
            
    return patch_labels

# ------------------------------------------------------------------
# 2. DATASET
# ------------------------------------------------------------------

class FLIRMultiLabelDataset(Dataset):
    """
    Dataset multi-label par token (patch) pour FLIR Aligned.
    mode  : 'rgb' | 'ir' | 'both'
    split : 'train' | 'val'
    """

    def __init__(self, root: str, split: str, mode: str, img_size: int = 448, patch_size: int = 16,
                 tokenizer_by_modality: bool = True, ir_mean: float = 0.0, ir_std: float = 1.0):
        super().__init__()
        assert mode  in ('rgb', 'ir', 'both', 'joint'), f"mode invalide : {mode}"
        assert split in ('train', 'val'),       f"split invalide : {split}"
        self.mode = mode
        self.patch_size = patch_size
        self.tokenizer_by_modality = tokenizer_by_modality

        root       = Path(root)
        img_dir    = root / 'AnnotatedImages'
        annot_dir  = root / 'Annotations'
        split_file = root / ('align_train.txt' if split == 'train' else 'align_validation.txt')

        self.transform_rgb = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(),
                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        if self.tokenizer_by_modality:
            # IR 1 canal -> audio_patch_embed. Normalisation = celle de rgb_ir_dataset à l'entraînement MJEPA
            self.transform_ir  = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(),
                                T.Normalize([ir_mean], [ir_std])])
            self.ir_pil_mode = 'L'
        else:
            self.transform_ir  = self.transform_rgb
            self.ir_pil_mode = 'RGB'

        with open(split_file, 'r') as f:
            prefixes = [l.strip() for l in f if l.strip()]

        self.samples = []
        skipped = 0

        for prefix in prefixes:
            base_name = prefix.replace('_PreviewData', '')

            rgb_path = img_dir  / f'{base_name}_RGB.jpg'
            ir_path  = img_dir  / f'{prefix}.jpeg'
            xml_path = annot_dir / f'{prefix}.xml'

            if mode in ('rgb', 'both', 'joint') and not rgb_path.exists():
                skipped += 1
                continue
            if mode in ('ir', 'both', 'joint') and not ir_path.exists():
                skipped += 1
                continue
            if not xml_path.exists():
                skipped += 1
                continue

            boxes_list = parse_xml_to_resized_boxes(str(xml_path), target_size=img_size)
            labels_np = generate_patch_labels(boxes_list, img_size=img_size, patch_size=self.patch_size, num_classes=4)
            
            self.samples.append((str(rgb_path), str(ir_path), labels_np))

        all_labels = np.stack([s[2] for s in self.samples])
        
        # Somme totale des patchs actifs pour chaque classe à travers tout le dataset
        class_counts = all_labels.sum(axis=(0, 1)).astype(int)
        
        print(f"[{split}] mode={mode} | {len(self.samples)} images "
              f"({skipped} ignorées) | "
              + " | ".join(f"{IDX2CLASS[i]}={class_counts[i]} patches"
                           for i in range(NUM_CLASSES)))

        # Fréquence globale calculée sur l'ensemble de TOUS les patchs du dataset
        freq = all_labels.mean(axis=(0, 1))
        
        # pos_weight de taille [4], indispensable pour équilibrer la BCEWithLogitsLoss
        self.pos_weight = torch.tensor((1.0 - freq) / (freq + 1e-8), dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, ir_path, label_tab = self.samples[idx]

        label_tensor = torch.tensor(label_tab, dtype=torch.float32)

        if self.mode == 'rgb':
            img = Image.open(rgb_path).convert('RGB')
            return self.transform_rgb(img), label_tensor

        elif self.mode == 'ir':
            img = Image.open(ir_path).convert(self.ir_pil_mode)
            return self.transform_ir(img), label_tensor

        else:   # both / joint
            rgb_img = Image.open(rgb_path).convert('RGB')
            ir_img  = Image.open(ir_path).convert(self.ir_pil_mode)
            return self.transform_rgb(rgb_img), self.transform_ir(ir_img), label_tensor


# ------------------------------------------------------------------
# 3. CHARGEMENT DE L'ENCODEUR LE MUMO
# ------------------------------------------------------------------

class LeMuMoEncoderWrapper(nn.Module):
    """Interface encoder(x, modality) attendue par le script.
       rgb -> RGB + IR à zéro ; ir -> RGB à zéro + IR ; joint(rgb, ir) -> les deux.
       Sortie : fusion tokens [B, 196, 384] après la norme finale du ViT."""
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.embed_dim = net.vit_embed_dim
        self.ir_channels = net.range_patch_embed.in_channels

    def _fusion_tokens(self, rgb, ir):
        # _forward_batch attend [B, V, C, H, W] ; V=1 vue
        _, _, fusion = self.net.forward_with_fusion_tokens(rgb.unsqueeze(1), ir.unsqueeze(1))
        return fusion

    def joint(self, rgb, ir):
        return self._fusion_tokens(rgb, ir)

    def forward(self, x, modality='rgb'):
        B, _, H, W = x.shape
        if modality == 'rgb':
            return self._fusion_tokens(x, x.new_zeros(B, self.ir_channels, H, W))
        elif modality == 'ir':
            return self._fusion_tokens(x.new_zeros(B, 3, H, W), x)
        raise ValueError(f"modality inconnue : {modality}")


def load_target_encoder(checkpoint_path: str, model_name: str, device: torch.device,
                        patch_size: int, crop_size: int):
    """Instancie MMEncoderC_FusionTokens avec la config sauvegardée par train.py
       (même construction que train.py) et charge ck['encoder']."""
    from src.encoder import MMEncoderC_FusionTokens

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {}) or {}
    assert cfg.get('fusion_tokens_sigreg', False), \
        f"{checkpoint_path} n'est pas un checkpoint Le MuMo fusion tokens (fusion_tokens_sigreg=False)"
    assert patch_size == 16, "Le MuMo utilise des patchs 16x16"
    vit_size = str(cfg.get('vit_size', 'small')).lower()
    if model_name and model_name != f'vit_{vit_size}':
        print(f"⚠️  --model_name={model_name} ignoré : le checkpoint est un ViT-{vit_size}")

    net = MMEncoderC_FusionTokens(
        proj_dim=int(cfg.get('proj_dim', 16)),
        img_size=crop_size,
        range_channels=1,
        aligned_mode=bool(cfg.get('aligned_mode', True)),
        vit_size=vit_size,
        attention_mode=str(cfg.get('fusion_tokens_variant', 'prune_after_first')).lower(),
        fusion_start_layer=int(cfg.get('fusion_start_layer', 0) or 0),
    )
    msg = net.load_state_dict(ckpt['encoder'], strict=False)
    assert not msg.unexpected_keys, f"clés inattendues : {msg.unexpected_keys}"
    assert not msg.missing_keys,    f"clés manquantes : {msg.missing_keys}"

    encoder = LeMuMoEncoderWrapper(net)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    encoder.to(device)
    print(f"✅ Le MuMo encoder chargé depuis {checkpoint_path} (run {ckpt.get('run_name')}, "
          f"variant={net.attention_mode}, ViT-{vit_size})")
    return encoder


# ------------------------------------------------------------------
# 4. LINEAR PROBE
# ------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x) # [B, N_patches, num_classes]


@torch.no_grad()
def extract_features(args, encoder, images: torch.Tensor, modality: str) -> torch.Tensor:
    """[B, C, 224, 224] -> [B, 196, D] tokens (dernier niveau, norms_block[-1])"""
    feats = encoder(images, modality=modality)
    # print(f"  🔍 Features extraites (shape={feats.shape})")
    _,n,_ = feats.shape
    if n == 197 or n == 785:  # Cas ViT avec token CLS
        feats = feats[:, 1:, :]  # [B, 197, 192] -> [B, 196, 192]
    return feats

def extract_and_save_features(args,encoder, loader, device, mode, save_path):
    if os.path.exists(save_path) and args.use_saved_features:
        print(f"  ⚡ Features déjà extraites trouvées : chargement depuis {save_path}")
        return torch.load(save_path)

    print(f"  ⚙️  Extraction des features pour {mode}...")
    encoder.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extraction"):
            if mode == 'both':
                rgb, ir, labels = batch
                rgb, ir = rgb.to(device), ir.to(device)
                f_rgb = extract_features(args = args, encoder = encoder, images = rgb, modality='rgb')
                f_ir  = extract_features(args = args, encoder = encoder, images = ir, modality='ir')
                features = torch.cat([f_rgb, f_ir], dim=-1)  # [B, n_patches, embed_dim*2]
            elif mode == 'joint':
                rgb, ir, labels = batch
                features = encoder.joint(rgb.to(device), ir.to(device))  # [B, n_patches, embed_dim]
            else:
                images, labels = batch
                images = images.to(device)
                features = extract_features(args = args, encoder = encoder, images = images, modality=mode)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    all_features = torch.cat(all_features, dim=0)
    all_labels   = torch.cat(all_labels, dim=0)

    data = {'features': all_features, 'labels': all_labels}
    torch.save(data, save_path)
    print(f"  💾 Sauvegardé dans {save_path} (Shape: {all_features.shape})")
    
    return data


# ------------------------------------------------------------------
# 5. BOUCLE TRAIN / VAL
# ------------------------------------------------------------------

def run_epoch(args,encoder, probe, loader, optimizer, criterion,
              device, mode, train: bool):
    probe.train(train)
    total_loss = 0.0
    total      = 0
    all_labels = []
    all_scores = []

    for batch in tqdm(loader, leave=False, desc='train' if train else 'val'):
        if mode == 'both':
            rgb, ir, labels = batch
            rgb    = rgb.to(device)
            ir     = ir.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                f_rgb = extract_features(args, encoder, rgb, modality='rgb')    # [B, N_patches, 192]
                f_ir  = extract_features(args, encoder, ir, modality='ir')     # [B, N_patches, 192]
            features = torch.cat([f_rgb, f_ir], dim=-1)   # [B, n_patches, embed_dim*2]
        elif mode == 'joint':
            rgb, ir, labels = batch
            labels = labels.to(device)
            with torch.no_grad():
                features = encoder.joint(rgb.to(device), ir.to(device))
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                if mode == 'rgb':
                    features = extract_features(args, encoder, images, modality='rgb')   # [B, N_patches, embed_dim]
                else:
                    features = extract_features(args, encoder, images, modality='ir')   # [B, N_patches, embed_dim]

        logits = probe(features)          # [B, N_patches, num_classes]
        loss   = criterion(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total      += labels.size(0)

        # Accumule pour le calcul du mAP
        all_labels.append(labels.cpu().numpy())
        all_scores.append(torch.sigmoid(logits).detach().cpu().numpy())

    avg_loss   = total_loss / total
    all_labels = np.concatenate(all_labels, axis=0) # [N_images, N_patches, 4]
    all_labels = all_labels.reshape(-1, NUM_CLASSES) # [N_images * N_patches, 4]

    all_scores = np.concatenate(all_scores, axis=0) # [N_images, N_patches, 4]
    all_scores = all_scores.reshape(-1, NUM_CLASSES) # [N_images * N_patches, 4]

    # AP par classe sur chaque token en gros 
    ap_per_class = {}
    for i, cls in IDX2CLASS.items():
        if all_labels[:, i].sum() == 0:
            ap_per_class[cls] = float('nan')
        else:
            ap_per_class[cls] = average_precision_score(
                all_labels[:, i], all_scores[:, i])

    valid_aps = [v for v in ap_per_class.values() if not np.isnan(v)]
    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0

    return avg_loss, mAP, ap_per_class, all_labels, all_scores

def run_epoch_features(probe, loader, optimizer, criterion, device, train: bool):
    probe.train(train)
    total_loss = 0.0
    total      = 0
    all_labels = []
    all_scores = []

    for features, labels in tqdm(loader, leave=False, desc='train' if train else 'val'):
        features = features.to(device)
        labels   = labels.to(device)

        logits = probe(features)          # [B, N_patches, num_classes]
        loss   = criterion(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total      += labels.size(0)

        all_labels.append(labels.cpu().numpy())
        all_scores.append(torch.sigmoid(logits).detach().cpu().numpy())

    avg_loss   = total_loss / total
    all_labels = np.concatenate(all_labels, axis=0) # [N_images, N_patches, 4]
    all_labels = all_labels.reshape(-1, NUM_CLASSES) # [N_images * N_patches, 4]
    
    all_scores = np.concatenate(all_scores, axis=0) # [N_images, N_patches, 4]
    all_scores = all_scores.reshape(-1, NUM_CLASSES) # [N_images * N_patches, 4] 

    ap_per_class = {}
    for i, cls in IDX2CLASS.items():
        if all_labels[:, i].sum() == 0:
            ap_per_class[cls] = float('nan')
        else:
            ap_per_class[cls] = average_precision_score(all_labels[:, i], all_scores[:, i])

    valid_aps = [v for v in ap_per_class.values() if not np.isnan(v)]
    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0

    return avg_loss, mAP, ap_per_class, all_labels, all_scores

def plot_roc_curves(train_labels, train_scores, val_labels, val_scores, mode, output_dir):
    fig = plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'gray']
    
    for i, cls_name in IDX2CLASS.items():
        # --- Validation Curve ---
        if val_labels[:, i].sum() > 0:
            fpr_val, tpr_val, _ = roc_curve(val_labels[:, i], val_scores[:, i])
            roc_auc_val = auc(fpr_val, tpr_val)
            plt.plot(fpr_val, tpr_val, color=colors[i], lw=2, 
                     label=f'Val {cls_name} (AUC = {roc_auc_val:.3f})')
        
        # --- Train Curve ---
        if train_labels[:, i].sum() > 0:
            fpr_train, tpr_train, _ = roc_curve(train_labels[:, i], train_scores[:, i])
            roc_auc_train = auc(fpr_train, tpr_train)
            plt.plot(fpr_train, tpr_train, color=colors[i], lw=2, linestyle='--', alpha=0.6,
                     label=f'Train {cls_name} (AUC = {roc_auc_train:.3f})')

    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Mode: {mode.upper()}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Save the plot
    save_path = os.path.join(output_dir, f'roc_curve_{mode}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Courbe ROC sauvegardée : {save_path}")

    return fig


# ------------------------------------------------------------------
# 6. ENTRAÎNEMENT D'UNE CONFIG
# ------------------------------------------------------------------

def train_probe(args, mode: str, writer: SummaryWriter) -> float:
    device = torch.device(args.device)
    tag    = mode.upper()
    if args.method == 'images':
        print(f"\n{'='*55}")
        print(f"  Linear Probing multi-label — mode : {tag}")
        print(f"{'='*55}")

        train_ds = FLIRMultiLabelDataset(args.flir_root, 'train', mode, tokenizer_by_modality=args.tokenizer_by_modality, ir_mean=args.ir_mean, ir_std=args.ir_std)
        val_ds   = FLIRMultiLabelDataset(args.flir_root, 'val',   mode, tokenizer_by_modality=args.tokenizer_by_modality, ir_mean=args.ir_mean, ir_std=args.ir_std)

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                num_workers=args.workers, pin_memory=True,
                                drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                                num_workers=args.workers, pin_memory=True)

        encoder = load_target_encoder(
            args.jepa_checkpoint, args.model_name, device,
            crop_size=args.crop_size, patch_size=args.patch_size)

    else : 
        print(f"\n{'='*55}")
        print(f"  Linear Probing OFFLINE — mode : {tag}")
        print(f"{'='*55}")

        train_ds = FLIRMultiLabelDataset(args.flir_root, 'train', mode, img_size=args.crop_size, tokenizer_by_modality=args.tokenizer_by_modality, ir_mean=args.ir_mean, ir_std=args.ir_std)
        val_ds   = FLIRMultiLabelDataset(args.flir_root, 'val',   mode, img_size=args.crop_size, tokenizer_by_modality=args.tokenizer_by_modality, ir_mean=args.ir_mean, ir_std=args.ir_std)

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.workers)

        encoder = load_target_encoder(
            args.jepa_checkpoint, args.model_name, device,
            crop_size=args.crop_size, patch_size=args.patch_size)

        suffix = f"_tbm{args.tokenizer_by_modality}"
        train_path = os.path.join(args.output_dir, f'{mode}_train_features{suffix}.pt')
        val_path   = os.path.join(args.output_dir, f'{mode}_val_features{suffix}.pt')

        train_data = extract_and_save_features(args, encoder, train_loader, device, mode, train_path)
        val_data   = extract_and_save_features(args,encoder, val_loader,   device, mode, val_path)

        del encoder
        torch.cuda.empty_cache()

        if args.seed is not None and args.seed >= 0:
            set_seed(args.seed)

        feat_train_ds = TensorDataset(train_data['features'], train_data['labels'])
        feat_val_ds   = TensorDataset(val_data['features'], val_data['labels'])

        feat_train_loader = DataLoader(feat_train_ds, batch_size=args.batch, shuffle=True)
        feat_val_loader   = DataLoader(feat_val_ds,   batch_size=args.batch, shuffle=False)

    if args.method == 'images':
        embed_dim = encoder.embed_dim
        actual_embed_dim = embed_dim * 2 if mode == 'both' else embed_dim
    else:
        actual_embed_dim = train_data['features'].shape[-1]

    probe = LinearProbe(actual_embed_dim, num_classes=NUM_CLASSES).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # BCEWithLogitsLoss pondérée — compense le déséquilibre car >> person >> bicycle
    criterion = nn.BCEWithLogitsLoss(pos_weight=train_ds.pos_weight.to(device))

    best_val_map = 0.0
    best_epoch_data = None

    for epoch in range(1, args.epochs + 1):
        if args.method == 'images':
            train_loss, train_map, train_ap, train_labels, train_scores = run_epoch(
                args, encoder, probe, train_loader, optimizer, criterion, device, mode, train=True)
            val_loss, val_map, val_ap, val_labels, val_scores = run_epoch(
                args, encoder, probe, val_loader, optimizer, criterion, device, mode, train=False)
        else:
            train_loss, train_map, train_ap, train_labels, train_scores = run_epoch_features(
                probe, feat_train_loader, optimizer, criterion, device, train=True)
            val_loss, val_map, val_ap, val_labels, val_scores = run_epoch_features(
                probe, feat_val_loader, optimizer, criterion, device, train=False)
            
        scheduler.step()

        # TensorBoard
        writer.add_scalar(f'loss/{tag}_train', train_loss, epoch)
        writer.add_scalar(f'loss/{tag}_val',   val_loss,   epoch)
        writer.add_scalar(f'mAP/{tag}_train',  train_map,  epoch)
        writer.add_scalar(f'mAP/{tag}_val',    val_map,    epoch)
        for cls in FLIR_CLASSES:
            if not np.isnan(val_ap[cls]):
                writer.add_scalar(f'AP_{tag}/{cls}_val', val_ap[cls], epoch)
        wandb_log(args.wandb_run, {
            f'{tag}/epoch': epoch,
            f'{tag}/train_loss': train_loss, f'{tag}/val_loss': val_loss,
            f'{tag}/train_mAP': train_map, f'{tag}/val_mAP': val_map,
            **{f'{tag}/val_AP_{cls}': val_ap[cls] for cls in FLIR_CLASSES if not np.isnan(val_ap[cls])},
        })

        # Print terminal
        ap_str = ' | '.join(f"{cls}={val_ap[cls]:.3f}"
                             for cls in FLIR_CLASSES
                             if not np.isnan(val_ap[cls]))
        print(f"[{tag}] {epoch:3d}/{args.epochs} | "
              f"train loss={train_loss:.4f} mAP={train_map:.3f} | "
              f"val loss={val_loss:.4f} mAP={val_map:.3f} | {ap_str}")

        if val_map > best_val_map:
            best_val_map = val_map
            save_path = os.path.join(args.output_dir, f'best_probe_{mode}.pth')
            torch.save(probe.state_dict(), save_path)
            print(f"  💾 Meilleur {tag} sauvegardé (val mAP={best_val_map:.3f})")

            best_epoch_data = {
                'train_labels': train_labels,
                'train_scores': train_scores,
                'val_labels': val_labels,
                'val_scores': val_scores
            }

    assert best_epoch_data is not None, "aucune époque avec val mAP > 0 : probe cassé ?"
    if best_epoch_data:
        fig = plot_roc_curves(
            best_epoch_data['train_labels'], best_epoch_data['train_scores'],
            best_epoch_data['val_labels'], best_epoch_data['val_scores'],
            mode, args.output_dir
        )
    writer.add_figure(f'ROC_Curves/{mode.upper()}', fig, global_step=args.epochs)
    if args.wandb_run is not None:
        import wandb
        args.wandb_run.log({f'{tag}/roc_curve': wandb.Image(fig)})
    plt.close(fig)

    print(f"\n✨ SCORES ROC FINAUX POUR {tag} (Meilleure Époque) :")
    for i, cls_name in IDX2CLASS.items():
        if best_epoch_data['val_labels'][:, i].sum() > 0:
            fpr_val, tpr_val, _ = roc_curve(best_epoch_data['val_labels'][:, i], best_epoch_data['val_scores'][:, i])
            roc_auc_val = auc(fpr_val, tpr_val)
            print(f"METRIC_AUC_{tag}_{cls_name.upper()}: {roc_auc_val:.4f}")
            if args.wandb_run is not None:
                args.wandb_run.summary[f'{tag}/best_val_AUC_{cls_name}'] = roc_auc_val
    if args.wandb_run is not None:
        args.wandb_run.summary[f'{tag}/best_val_mAP'] = best_val_map

    return best_val_map


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected')

# ------------------------------------------------------------------
# 7. MAIN
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Linear Probing Le MuMo JEPA multi-label per-token sur FLIR Aligned',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--seed', type=int, default=42, help="Graine aléatoire pour reproductibilité")

    # JEPA
    parser.add_argument('--jepa_checkpoint', type=str, required=True)
    parser.add_argument('--model_name',   type=str, default='vit_small')
    parser.add_argument('--crop_size',    type=int, default=224)
    parser.add_argument('--patch_size',   type=int, default=16)
    parser.add_argument('--tokenizer_by_modality', type=str2bool, default=True,
                    help="Le MuMo : toujours True (IR 1 canal -> range_patch_embed)")
    parser.add_argument('--ir_mean', type=float, default=0.449,
                    help="Normalisation IR = THERMAL_MEAN de src/flir_dataset.py (entraînement Le MuMo)")
    parser.add_argument('--ir_std',  type=float, default=0.226,
                    help="idem (THERMAL_STD)")


    # Dataset
    parser.add_argument('--flir_root', type=str, required=True)
    parser.add_argument('--method', type=str, default='images', choices=['images', 'features'])
    parser.add_argument('--use_saved_features', type=str2bool, default=True,
                        help="Si True, réutilise les features extraites précédemment (si elles existent) au lieu de les recalculer à chaque run.")

    # Training
    parser.add_argument('--epochs',  type=int,   default=30)
    parser.add_argument('--batch',   type=int,   default=128)
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--wd',      type=float, default=1e-4)
    parser.add_argument('--workers', type=int,   default=16)
    parser.add_argument('--device',  type=str,   default='cuda:0')

    # Modes
    parser.add_argument('--modes', nargs='+', default=['rgb', 'ir', 'both'],
                        choices=['rgb', 'ir', 'both', 'joint'])

    # Output
    parser.add_argument('--output_dir', type=str, default='./output_linear_probing')
    parser.add_argument('--log_dir',    type=str, default='./logs_linear_probing')
    parser.add_argument('--wandb_project', type=str, default='',
                        help="Projet W&B ('' = pas de W&B)")
    parser.add_argument('--wandb_group', type=str, default='',
                        help="Groupe W&B (défaut : nom du dossier du checkpoint = run de pretraining)")

    args = parser.parse_args()
    if not args.tokenizer_by_modality:
        parser.error("Le MuMo a un patch-embed IR 1 canal : --tokenizer_by_modality doit rester True")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir,    exist_ok=True)

    logdir = os.path.join(args.log_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    writer = SummaryWriter(log_dir=logdir)
    args.wandb_run = wandb_init(args)

    results = {}
    for mode in args.modes:
        results[mode] = train_probe(args, mode, writer)

    writer.close()
    if args.wandb_run is not None:
        args.wandb_run.finish()

    # ------------------------------------------------------------------
    # Résumé final
    # ------------------------------------------------------------------
    print('\n' + '='*55)
    print('  RÉSUMÉ FINAL — Linear Probing JEPA sur FLIR Aligned')
    print('='*55)
    for mode, map_score in results.items():
        bar = '█' * int(map_score * 30)
        print(f"  {mode.upper():5s} │ {bar:<30s} │ mAP={map_score:.3f}")
    print('='*55)

    if 'rgb' in results and 'ir' in results:
        gap = abs(results['rgb'] - results['ir'])
        if gap < 0.05:
            print("\n✅ mAP RGB ≈ mAP IR (gap < 5%)"
                  " → le target_encoder aligne bien les deux modalités.")
        else:
            worse = 'IR' if results['rgb'] > results['ir'] else 'RGB'
            print(f"\n⚠️  Gap RGB/IR = {gap:.3f}"
                  f" → la modalité {worse} est moins bien représentée.")

    if 'both' in results and ('rgb' in results or 'ir' in results):
        best_single = max(results.get('rgb', 0), results.get('ir', 0))
        gain = results['both'] - best_single
        if gain < 0.02:
            print(f"✅ La fusion RGB+IR n'apporte pas grand chose (+{gain:.3f})"
                  " → les deux modalités codent la même sémantique (alignement réussi).")
        else:
            print(f"ℹ️  La fusion apporte +{gain:.3f}"
                  " → les modalités restent complémentaires.")


if __name__ == '__main__':
    main()