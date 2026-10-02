#!/bin/bash
# =============================================================================
#  Le MuMo JEPA ViT-S/16 @224 — pretraining SSL sur KAIST COMPLET (1 frame sur 4) (RGB/IR)
#  puis probing FLIR à encodeur gelé (protocole "Waymo->FLIR frozen transfer"
#  du papier, avec KAIST à la place de Waymo).
#
#    1) pretraining : imageSets/train-all-04.txt + test-all-04.txt (toutes les
#       séquences, 1 frame sur 4), encoder_only_mode, recette papier
#       (fusion tokens pruned + SIGReg sur le CLS joint, bs 64, lr 1e-4,
#       V=2 globales + 8 locales, lamb 0.1, 20 epochs = schedule FLIR scratch).
#    2) probing FLIR : encodeur gelé, têtes CenterNet 2D + occupancy,
#       V=1, pas de crops locaux, probes à 640x640, 5 epochs.
#       Éval en RGB+IR ("val"), RGB seul ("rgb_only") et IR seul ("lidar_only").
#
#  Lancer depuis la racine du repo :  sbatch slurm/lemumo_kaist.sl
#  Refaire seulement le probing     :  sbatch --export=ALL,SKIP_PRETRAIN=1 slurm/lemumo_kaist.sl
#
#  Cluster : CRIANN par défaut. Pour Jean Zay, commenter le bloc CRIANN,
#  décommenter le bloc Jean Zay (retirer un '#') et renseigner le compte.
# =============================================================================
#SBATCH --job-name=lemumo_kaist
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
# ---- CRIANN
#SBATCH --partition=gpu_h200
# ---- Jean Zay (H100)
##SBATCH --account=XXX@h100
##SBATCH -C h100
##SBATCH --qos=qos_gpu_h100-t3
##SBATCH --hint=nomultithread

set -euo pipefail

# ---------------------------------------------------------------- cluster
if [ -z "${CLUSTER:-}" ]; then
    if [ -n "${IDRPROJ:-}" ] || [[ "${SLURM_CLUSTER_NAME:-}" == *jean* ]]; then
        CLUSTER=jeanzay
    else
        CLUSTER=criann
    fi
fi

module purge
if [ "${CLUSTER}" = "jeanzay" ]; then
    module load pytorch-gpu/py3/2.6.0
    DATA_DIR="${DATA_DIR:-${SCRATCH}/datasets}"
    LOG_DIR="${LOG_DIR:-${WORK}/logs/lemumo}"
    export WANDB_MODE=offline            # pas d'internet sur les nœuds de calcul
else
    module load cray-python/3.11.7
    module load aidl/pytorch/2.6.0-cuda12.6
    DATA_DIR="${DATA_DIR:-/home/2023029/PARTAGE/datasets}"
    LOG_DIR="${LOG_DIR:-/home/2023029/PARTAGE/abelgh02/logs/lemumo}"
    export WANDB_MODE="${WANDB_MODE:-online}"
fi

# ---------------------------------------------------------------- chemins
KAIST_ROOT="${DATA_DIR}/KAIST"
# Probing Le MuMo = FLIR ADAS v2 (format COCO : images_rgb_train/coco.json,
# index.json, rgb_to_thermal_vid_map.json). Le "FLIR aligned" (JPEGImages +
# align_*.txt + XML VOC) utilisé par MJEPA n'est PAS ce format.
FLIR_ROOT="${FLIR_ROOT:-${DATA_DIR}/FLIR_ADAS_v2}"
# Splits KAIST (imageSets/), séparés par des virgules. Si test-all-04.txt
# n'existe pas chez toi il est ignoré avec un warning (seul train est utilisé).
KAIST_SPLITS="${KAIST_SPLITS:-train-all-04.txt,test-all-04.txt}"

# ---------------------------------------------------------------- hyperparamètres (défauts papier)
EPOCHS=20
BATCH_SIZE=64
LR=1.0e-4
LAMB=0.1
NUM_LOCAL=8
PROBE_EPOCHS=5
PROBE_IMG_SIZE=640
NUM_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 2 ))

RUN_NAME="lemumo_vits16_kaistfull04_ep${EPOCHS}_bs${BATCH_SIZE}_lamb${LAMB}"
PROBE_RUN_NAME="${RUN_NAME}_flirprobe_ep${PROBE_EPOCHS}_res${PROBE_IMG_SIZE}"
CKPT="${LOG_DIR}/${RUN_NAME}/latest.pt"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTHONUNBUFFERED=1
export WANDB_DIR="${LOG_DIR}"

echo "================================================================="
echo " Job ${SLURM_JOB_ID} | cluster=${CLUSTER} | node ${SLURM_JOB_NODELIST}"
echo " KAIST     : ${KAIST_ROOT}  (${KAIST_SPLITS})"
echo " FLIR      : ${FLIR_ROOT}"
echo " Run       : ${LOG_DIR}/${RUN_NAME}"
echo " pretrain  : epochs=${EPOCHS} bs=${BATCH_SIZE} lr=${LR} lamb=${LAMB} local=${NUM_LOCAL}"
echo " probing   : epochs=${PROBE_EPOCHS} res=${PROBE_IMG_SIZE}"
echo "================================================================="
nvidia-smi

# ---------------------------------------------------------------- 1) pretraining SSL LLVIP
if [ "${SKIP_PRETRAIN:-0}" = "1" ] && [ -e "${CKPT}" ]; then
    echo "SKIP_PRETRAIN=1 : réutilise ${CKPT}"
else
    srun python3 train.py \
        dataset=kaist \
        +kaist_dataroot="${KAIST_ROOT}" \
        "+kaist_splits=[${KAIST_SPLITS}]" \
        +encoder_only_mode=true \
        epochs=${EPOCHS} \
        bs=${BATCH_SIZE} \
        lr=${LR} \
        lamb=${LAMB} \
        local_crops_number=${NUM_LOCAL} \
        +num_workers=${NUM_WORKERS} \
        +run_name="${RUN_NAME}" \
        +save_root="${LOG_DIR}" \
        hydra.run.dir="${LOG_DIR}/${RUN_NAME}/hydra_pretrain"
fi
test -e "${CKPT}" || { echo "ERREUR: checkpoint introuvable: ${CKPT}"; exit 1; }

# ---------------------------------------------------------------- 2) probing FLIR (encodeur gelé)
if [ ! -d "${FLIR_ROOT}/images_rgb_train" ] && [ ! -d "${FLIR_ROOT}/FLIR_ADAS_v2/images_rgb_train" ]; then
    echo "FLIR ADAS v2 introuvable sous ${FLIR_ROOT} : probing sauté (checkpoint : ${CKPT})."
    exit 0
fi
srun python3 train.py \
    dataset=flir \
    flir_dataroot="${FLIR_ROOT}" \
    +pretrained_encoder_path="${CKPT}" \
    +probe_only_training=true \
    V=1 \
    local_crops_number=0 \
    epochs=${PROBE_EPOCHS} \
    bs=${BATCH_SIZE} \
    lr=${LR} \
    +probe_lr=1.0e-3 \
    +patch_probe_lr=1.0e-3 \
    +probe_img_size=${PROBE_IMG_SIZE} \
    +num_workers=${NUM_WORKERS} \
    +run_name="${PROBE_RUN_NAME}" \
    +save_root="${LOG_DIR}" \
    hydra.run.dir="${LOG_DIR}/${PROBE_RUN_NAME}/hydra_probe"
