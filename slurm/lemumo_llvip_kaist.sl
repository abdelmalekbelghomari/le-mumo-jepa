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
#  Chaîne de reprise automatique (runs plus longs que la limite de 24h / 20h) :
#    - train.py sauvegarde ${RUN_DIR}/resume.pt à chaque epoch (encodeur,
#      optimiseur, scheduler LR, GradScaler, RNG) et reprend dessus.
#    - tant que latest.pt (checkpoint final) n'existe pas, chaque job soumet
#      son successeur (--dependency=afterany) AVANT d'entraîner ; si SLURM tue
#      le job à la limite de temps, le successeur reprend à l'epoch suivante.
#    - une fois latest.pt écrit, le job fait le probing puis annule son
#      successeur. MAX_CHAIN borne le nombre de jobs.
#    ~2.7 min/epoch sur H200 (mesuré à 20 epochs) => 600 epochs ≈ 27h ≈ 2 jobs.
#
#  Lancer depuis la racine du repo :  sbatch slurm/lemumo_llvip_kaist.sl
#  Autre nombre d'epochs           :  sbatch --export=ALL,EPOCHS=300 slurm/lemumo_llvip_kaist.sl
#  Refaire le probing              :  sbatch --export=ALL,FORCE_PROBE=1 slurm/lemumo_llvip_kaist.sl
#  Arrêter la chaîne               :  scancel du job en attente (dépendance afterany)
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
    export WANDB_MODE=offline            # pas d'internet : 'wandb sync ${WANDB_DIR}/wandb/offline-run-*' depuis une frontale
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
EPOCHS="${EPOCHS:-600}"            # comme MJEPA (mjepa_vits16_..._ep600)
BATCH_SIZE=64
LR=1.0e-4
LAMB=0.1
NUM_LOCAL=8
NUM_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 2 ))

# ---------------------------------------------------------------- chaîne de reprise
MAX_CHAIN="${MAX_CHAIN:-6}"          # nb max de jobs bout à bout
CHAIN_IDX="${CHAIN_IDX:-1}"          # maillon courant, propagé via --export
# chemin du .sl d'origine ($0 pointe sur la copie du spool SLURM)
SCRIPT_PATH="${SCRIPT_PATH:-$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | sed -n 's/^ *Command=//p' | head -n1)}"
SCRIPT_PATH="${SCRIPT_PATH:-$0}"

# ---------------------------------------------------------------- probing (identique aux autres benchs)
NUM_RUNS=10
CROP_SIZE=224
IR_MEAN=0.449            # THERMAL_MEAN/STD de src/flir_dataset.py = normalisation IR du pretraining
IR_STD=0.226
PROBE_MODES="${PROBE_MODES:-rgb ir both joint}"   # joint = RGB+IR ensemble dans l'encodeur (fusion Le MuMo)

RUN_NAME="lemumo_vits16_llvip-kaistfull04_ep${EPOCHS}_bs${BATCH_SIZE}_lamb${LAMB}"
RUN_DIR="${LOG_DIR}/${RUN_NAME}"
CKPT="${RUN_DIR}/latest.pt"           # écrit uniquement à la fin du pretraining
RESUME_CKPT="${RUN_DIR}/resume.pt"   # écrit à chaque epoch
PROBE_DIR="${RUN_DIR}/probings_latest_runs${NUM_RUNS}_cropsize${CROP_SIZE}"
PROBE_CSV="${PROBE_DIR}/probings_results_${RUN_NAME}.csv"
WANDB_RUN_ID="${RUN_NAME//./p}"      # id stable => une seule courbe W&B sur tous les maillons
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTHONUNBUFFERED=1
export WANDB_DIR="${LOG_DIR}"
# W&B : pretraining (loss, sigreg, inv, débit, VRAM par batch) puis probing
# (un run par seed + un run "summary" moyenne ± std), tous dans le groupe ${RUN_NAME}.
# WANDB_MODE=disabled pour couper W&B.
export WANDB_PROJECT="${WANDB_PROJECT:-lemumo_rgbir_small}"

echo "================================================================="
echo " Job ${SLURM_JOB_ID} | maillon ${CHAIN_IDX}/${MAX_CHAIN} | cluster=${CLUSTER} | node ${SLURM_JOB_NODELIST}"
echo " LLVIP     : ${LLVIP_ROOT}  (train + test)"
echo " KAIST     : ${KAIST_ROOT}  (${KAIST_SPLITS})"
echo " FLIR al.  : ${FLIR_ALIGNED_ROOT}"
echo " Run       : ${RUN_DIR}"
echo " pretrain  : epochs=${EPOCHS} bs=${BATCH_SIZE} lr=${LR} lamb=${LAMB} local=${NUM_LOCAL}"
echo " probing   : ${NUM_RUNS} runs | modes=${PROBE_MODES} | IR norm mean=${IR_MEAN} std=${IR_STD}"
echo " W&B       : project=${WANDB_PROJECT} group=${RUN_NAME} mode=${WANDB_MODE}"
echo "================================================================="
nvidia-smi

# ---------------------------------------------------------------- 1) pretraining SSL LLVIP + KAIST
NEXT_ID=""
cancel_successor() {
    if [ -n "${NEXT_ID}" ]; then
        scancel "${NEXT_ID}" && echo "Successeur ${NEXT_ID} annulé."
    fi
}

if [ -e "${CKPT}" ]; then
    echo "Pretraining terminé (${CKPT})."
else
    if [ -e "${RESUME_CKPT}" ]; then
        echo "Reprise depuis ${RESUME_CKPT} : epoch $(python3 -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "${RESUME_CKPT}" 2>/dev/null || echo '?')/${EPOCHS} déjà faites"
    fi
    # successeur soumis AVANT d'entraîner : il prend le relais si ce job est tué à la limite de temps
    if [ "${CHAIN_IDX}" -lt "${MAX_CHAIN}" ]; then
        NEXT_ID=$(sbatch --parsable \
            --dependency=afterany:${SLURM_JOB_ID} \
            --chdir="${SLURM_SUBMIT_DIR:-$(pwd)}" \
            --export=ALL,CHAIN_IDX=$((CHAIN_IDX + 1)),MAX_CHAIN=${MAX_CHAIN},EPOCHS=${EPOCHS},SCRIPT_PATH=${SCRIPT_PATH} \
            "${SCRIPT_PATH}")
        echo "Maillon ${CHAIN_IDX}/${MAX_CHAIN} : successeur soumis (job ${NEXT_ID}, afterany:${SLURM_JOB_ID})"
    else
        echo "Maillon ${CHAIN_IDX}/${MAX_CHAIN} : dernier maillon, pas de successeur."
    fi

    if ! srun python3 train.py \
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
        +ckpt_every_epochs=1 \
        +resume=true \
        +run_name="${RUN_NAME}" \
        +wandb_project="${WANDB_PROJECT}" \
        +wandb_run_id="${WANDB_RUN_ID}" \
        +save_root="${LOG_DIR}" \
        hydra.run.dir="${RUN_DIR}/hydra_pretrain_job${SLURM_JOB_ID}"; then
        # vraie erreur (un dépassement de temps tue le script avant d'arriver ici) : on casse la chaîne
        echo "ERREUR: train.py a échoué, arrêt de la chaîne."
        cancel_successor
        exit 1
    fi
fi
test -e "${CKPT}" || { echo "ERREUR: checkpoint introuvable: ${CKPT}"; cancel_successor; exit 1; }

# ---------------------------------------------------------------- 2) linear probing FLIR aligned
if [ -e "${PROBE_CSV}" ] && [ "${FORCE_PROBE:-0}" != "1" ]; then
    echo "Probing déjà fait (${PROBE_CSV}), FORCE_PROBE=1 pour le refaire."
else
    mkdir -p "${PROBE_DIR}"
    srun python3 probing/run_probings_lemumo.py \
        --amount_runs ${NUM_RUNS} \
        --jepa_checkpoint "${CKPT}" \
        --crop_size ${CROP_SIZE} \
        --flir_root "${FLIR_ALIGNED_ROOT}" \
        --output_dir "${PROBE_DIR}" \
        --csv_output "${PROBE_CSV}" \
        --log_dir "${PROBE_DIR}/tb" \
        --device "${PROBE_DEVICE:-cuda:0}" \
        --ir_mean ${IR_MEAN} \
        --ir_std ${IR_STD} \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_group "${RUN_NAME}" \
        --modes ${PROBE_MODES}
fi
cancel_successor
echo "Chaîne terminée."
