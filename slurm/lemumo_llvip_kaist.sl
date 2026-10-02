#!/bin/bash
# =============================================================================
#  Le MuMo JEPA ViT-S/16 @224 — pretraining SSL sur LLVIP + KAIST concaténés,
#  puis linear probing per-token sur FLIR aligned (protocole 4-JEPA / MJEPA).
#
#    1) pretraining : une seule liste de paires = ConcatDataset de rgb_ir.py
#         LLVIP : visible/{train,test} + infrared/{train,test}
#         KAIST : imageSets/train-all-04.txt + test-all-04.txt
#       encoder_only_mode, recette papier (fusion tokens pruned + SIGReg sur
#       le CLS joint, bs 64, lr 1e-4, V=2 globales + 8 locales, lamb 0.1).
#    2) probing : probing/run_probings_lemumo.py = copie de run_probings_mjepa.py
#       (10 runs, 30 ep, lr 1e-3, seeds 43..52, features cachées), seul le
#       chargement de l'encodeur change. Modes RGB / IR / BOTH (+ JOINT).
#       Résultats : ${RUN_DIR}/probings_*/probings_results_*.csv
#
#  Lancer depuis la racine du repo :  sbatch slurm/lemumo_llvip_kaist.sl
#  Refaire seulement le probing     :  sbatch --export=ALL,SKIP_PRETRAIN=1 slurm/lemumo_llvip_kaist.sl
#
#  Cluster : CRIANN par défaut. Pour Jean Zay, commenter le bloc CRIANN,
#  décommenter le bloc Jean Zay (retirer un '#') et renseigner le compte.
# =============================================================================
#SBATCH --job-name=lemumo_llvip_kaist
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
# ---- CRIANN
#SBATCH --partition=gpu_h200
# ---- Jean Zay (H100, qos t3 = 20h max : baisser --time à 20:00:00)
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
LLVIP_ROOT="${DATA_DIR}/LLVIP"
KAIST_ROOT="${DATA_DIR}/KAIST"
FLIR_ALIGNED_ROOT="${FLIR_ALIGNED_ROOT:-${DATA_DIR}/FLIR}"   # AnnotatedImages + Annotations + align_*.txt
KAIST_SPLITS="${KAIST_SPLITS:-train-all-04.txt,test-all-04.txt}"

# ---------------------------------------------------------------- hyperparamètres pretraining (défauts papier)
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE=64
LR=1.0e-4
LAMB=0.1
NUM_LOCAL=8
NUM_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 2 ))

# ---------------------------------------------------------------- probing (identique aux autres benchs)
NUM_RUNS=10
CROP_SIZE=224
IR_MEAN=0.449            # THERMAL_MEAN/STD de src/flir_dataset.py = normalisation IR du pretraining
IR_STD=0.226
PROBE_MODES="${PROBE_MODES:-rgb ir both joint}"   # joint = RGB+IR ensemble dans l'encodeur (fusion Le MuMo)

RUN_NAME="lemumo_vits16_llvip-kaistfull04_ep${EPOCHS}_bs${BATCH_SIZE}_lamb${LAMB}"
RUN_DIR="${LOG_DIR}/${RUN_NAME}"
CKPT="${RUN_DIR}/latest.pt"
PROBE_DIR="${RUN_DIR}/probings_latest_runs${NUM_RUNS}_cropsize${CROP_SIZE}"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTHONUNBUFFERED=1
export WANDB_DIR="${LOG_DIR}"

echo "================================================================="
echo " Job ${SLURM_JOB_ID} | cluster=${CLUSTER} | node ${SLURM_JOB_NODELIST}"
echo " LLVIP     : ${LLVIP_ROOT}  (train + test)"
echo " KAIST     : ${KAIST_ROOT}  (${KAIST_SPLITS})"
echo " FLIR al.  : ${FLIR_ALIGNED_ROOT}"
echo " Run       : ${RUN_DIR}"
echo " pretrain  : epochs=${EPOCHS} bs=${BATCH_SIZE} lr=${LR} lamb=${LAMB} local=${NUM_LOCAL}"
echo " probing   : ${NUM_RUNS} runs | modes=${PROBE_MODES} | IR norm mean=${IR_MEAN} std=${IR_STD}"
echo "================================================================="
nvidia-smi

# ---------------------------------------------------------------- 1) pretraining SSL LLVIP + KAIST
if [ "${SKIP_PRETRAIN:-0}" = "1" ] && [ -e "${CKPT}" ]; then
    echo "SKIP_PRETRAIN=1 : réutilise ${CKPT}"
else
    srun python3 train.py \
        dataset=rgbir \
        "+rgbir_sources=[llvip,kaist]" \
        +llvip_dataroot="${LLVIP_ROOT}" \
        "+llvip_splits=[train,test]" \
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
        hydra.run.dir="${RUN_DIR}/hydra_pretrain"
fi
test -e "${CKPT}" || { echo "ERREUR: checkpoint introuvable: ${CKPT}"; exit 1; }

# ---------------------------------------------------------------- 2) linear probing FLIR aligned
mkdir -p "${PROBE_DIR}"
srun python3 probing/run_probings_lemumo.py \
    --amount_runs ${NUM_RUNS} \
    --jepa_checkpoint "${CKPT}" \
    --crop_size ${CROP_SIZE} \
    --flir_root "${FLIR_ALIGNED_ROOT}" \
    --output_dir "${PROBE_DIR}" \
    --csv_output "${PROBE_DIR}/probings_results_${RUN_NAME}.csv" \
    --log_dir "${PROBE_DIR}/tb" \
    --device "${PROBE_DEVICE:-cuda:0}" \
    --ir_mean ${IR_MEAN} \
    --ir_std ${IR_STD} \
    --modes ${PROBE_MODES}
