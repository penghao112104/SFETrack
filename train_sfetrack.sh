#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

CONFIG=${CONFIG:-rgbt}
SAVE_DIR=${SAVE_DIR:-./output}
DATA_DIR=${DATA_DIR:-}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
SEED=${SEED:-0}
MOE_STAGE_START_EPOCH=${MOE_STAGE_START_EPOCH:-31}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-40}
RESUME=${RESUME:-0}

if [ -z "$DATA_DIR" ]; then
  echo "[error] DATA_DIR is required." >&2
  echo "Usage: DATA_DIR=/path/to/LasHeR CUDA_VISIBLE_DEVICES=0,1 bash train_sfetrack.sh" >&2
  exit 1
fi

if (( MOE_STAGE_START_EPOCH < 1 || MOE_STAGE_START_EPOCH > TOTAL_EPOCHS )); then
  echo "[error] MOE_STAGE_START_EPOCH must be between 1 and TOTAL_EPOCHS." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE=${NPROC_PER_NODE:-${#gpu_ids[@]}}
if [ "$NPROC_PER_NODE" -lt 1 ]; then
  echo "[error] NPROC_PER_NODE must be at least 1." >&2
  exit 1
fi
if [ "$NPROC_PER_NODE" -eq 1 ]; then
  MODE=${MODE:-single}
else
  MODE=${MODE:-multiple}
fi

mkdir -p "$SAVE_DIR"
SAVE_DIR_ABS=$(cd "$SAVE_DIR" && pwd)
LOG_DIR="$SAVE_DIR_ABS/logs"
mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-"$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"}
exec > >(tee -a "$LOG_FILE") 2>&1

python tracking/create_default_local_file.py \
  --workspace_dir "$PROJECT_ROOT" \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR_ABS"

train_cmd=(python tracking/train.py
  --script bat
  --config "$CONFIG"
  --save_dir "$SAVE_DIR_ABS"
  --mode "$MODE"
  --nproc_per_node "$NPROC_PER_NODE"
  --seed "$SEED"
  --cfg_override "TRAIN.EPOCH=$TOTAL_EPOCHS"
  --cfg_override "TEST.EPOCH=$TOTAL_EPOCHS"
  --cfg_override "MODEL.MOE.ENABLE=True"
  --cfg_override "TRAIN.MOE_STAGE_START_EPOCH=$MOE_STAGE_START_EPOCH"
  --cfg_override "TRAIN.MOE_STAGE_ALL_MOE_END=$TOTAL_EPOCHS")

if [ "$RESUME" != "1" ] && [ "$RESUME" != "true" ] && [ "$RESUME" != "True" ]; then
  train_cmd+=(--no_resume)
fi

echo "[train] dataset: $DATA_DIR"
echo "[train] GPUs: $CUDA_VISIBLE_DEVICES"
echo "[train] log: $LOG_FILE"
echo "[train] schedule: 1-$((MOE_STAGE_START_EPOCH - 1)) without MoE, $MOE_STAGE_START_EPOCH-$TOTAL_EPOCHS with MoE"
"${train_cmd[@]}"
