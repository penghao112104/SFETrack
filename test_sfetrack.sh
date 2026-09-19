#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

CHECKPOINT=${SFETRACK_CHECKPOINT:-${CTETRACK_CHECKPOINT:-${BAT_CHECKPOINT:-${TEST_CHECKPOINT:-}}}}
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  CHECKPOINT=$1
  shift
fi

YAML_NAME=${YAML_NAME:-rgbt}
DATASET_NAME=${DATASET_NAME:-LasHeR}
SEQ_HOME=${SEQ_HOME:-}
RESULT_ROOT=${RESULT_ROOT:-./RGBT_workspace/results}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TEST_EPOCH=${TEST_EPOCH:-40}

if [ -z "$SEQ_HOME" ]; then
  echo "[error] SEQ_HOME is required." >&2
  echo "Usage: SEQ_HOME=/path/to/dataset bash test_sfetrack.sh /path/to/checkpoint.pth.tar" >&2
  exit 1
fi

if [ -z "$CHECKPOINT" ]; then
  CHECKPOINT=$(find "$PROJECT_ROOT/output/checkpoints" -maxdepth 1 -type f \
    -name '*Track_ep*.pth.tar' 2>/dev/null | sort -V | tail -n 1 || true)
fi

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
  echo "[error] checkpoint not found: ${CHECKPOINT:-<empty>}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES
SFETRACK_CHECKPOINT=$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")
mkdir -p "$RESULT_ROOT"
RGBT_RESULT_ROOT=$(cd "$RESULT_ROOT" && pwd)
export SFETRACK_CHECKPOINT
export BAT_CHECKPOINT="$SFETRACK_CHECKPOINT"
export RGBT_RESULT_ROOT
export RGBT_SAVE_NAME=${RGBT_SAVE_NAME:-SFETrack}

IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${NUM_GPUS:-${#gpu_ids[@]}}
THREADS=${THREADS:-$NUM_GPUS}

python tracking/create_default_local_file.py \
  --workspace_dir "$PROJECT_ROOT" \
  --data_dir "${DATA_DIR:-$SEQ_HOME}" \
  --save_dir "$PROJECT_ROOT/output"

echo "[test] dataset: $DATASET_NAME"
echo "[test] dataset root: $SEQ_HOME"
echo "[test] checkpoint: $SFETRACK_CHECKPOINT"
echo "[test] result root: $RGBT_RESULT_ROOT"

python RGBT_workspace/test_rgbt_mgpus.py \
  --script_name bat \
  --dataset_name "$DATASET_NAME" \
  --seq_home "$SEQ_HOME" \
  --yaml_name "$YAML_NAME" \
  --num_gpus "$NUM_GPUS" \
  --threads "$THREADS" \
  --epoch "$TEST_EPOCH" \
  "$@"
