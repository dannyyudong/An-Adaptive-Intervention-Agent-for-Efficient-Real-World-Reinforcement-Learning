#!/usr/bin/env bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
EXP_DIR=$SCRIPT_DIR
RUN_ID_FILE=${RUN_ID_FILE:-$EXP_DIR/.latest_run_id}
cd "$EXP_DIR"

for required_name in UR_ROBOT_IP HAND_SERIAL EXTERNAL_SERIAL; do
    if [[ -z "${!required_name:-}" ]]; then
        echo "Set $required_name explicitly before evaluation." >&2
        exit 1
    fi
done

if [[ ( "${LEARNED_OPTION_SCHEDULER:-0}" == "1" || "${MANUAL_OPTION_SCHEDULER:-1}" == "1" ) \
      && -z "${CODE_POLICY_USB_TARGET:-}" ]]; then
    echo "Set CODE_POLICY_USB_TARGET='x y z' after calibrating the USB target." >&2
    exit 1
fi

if [[ -z "${RUN_ID:-}" ]]; then
    if [[ -f "$RUN_ID_FILE" ]]; then
        RUN_ID=$(<"$RUN_ID_FILE")
    else
        echo "Set RUN_ID before evaluation." >&2
        exit 1
    fi
fi

CHECKPOINT_PATH=${CHECKPOINT_PATH:-$EXP_DIR/aia_usb_insert_${RUN_ID}}
if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "CHECKPOINT_PATH does not exist: $CHECKPOINT_PATH" >&2
    exit 1
fi

if [[ -z "${EVAL_CHECKPOINT_STEP:-}" ]]; then
    latest=$(find "$CHECKPOINT_PATH" -maxdepth 1 -type d -name 'checkpoint_*' -printf '%f\n' | sed 's/^checkpoint_//' | sort -n | tail -1)
    if [[ -z "$latest" ]]; then
        echo "No checkpoint_* directory found in $CHECKPOINT_PATH" >&2
        exit 1
    fi
    EVAL_CHECKPOINT_STEP=$latest
fi

EVAL_N_TRAJS=${EVAL_N_TRAJS:-5}
SAVE_VIDEO=${SAVE_VIDEO:-1}
export VIDEO_RECORD_CAMERA_KEY=${VIDEO_RECORD_CAMERA_KEY:-external,wrist}
export VIDEO_RECORD_RAW=${VIDEO_RECORD_RAW:-1}
export VIDEO_RECORD_DIR=${VIDEO_RECORD_DIR:-$EXP_DIR/videos}
export VIDEO_RECORD_FPS=${VIDEO_RECORD_FPS:-10}

DEFAULT_VENV=${VENV:-$ROOT_DIR/.venv}
NVIDIA_LD_LIBRARY_PATH=$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cudnn/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cublas/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cusparse/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cusolver/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/nvjitlink/lib
export VENV=$DEFAULT_VENV
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH=$NVIDIA_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$NVIDIA_LD_LIBRARY_PATH
fi

args=(
    "$ROOT_DIR/examples/train_aia.py"
    --exp_name=aia_usb_insert
    --checkpoint_path="$CHECKPOINT_PATH"
    --actor
    --allow_existing_checkpoint_path
    --eval_checkpoint_step="$EVAL_CHECKPOINT_STEP"
    --eval_n_trajs="$EVAL_N_TRAJS"
    --debug
)

if [[ "${LEARNED_OPTION_SCHEDULER:-0}" == "1" ]]; then
    args+=(--learned_option_scheduler)
elif [[ "${MANUAL_OPTION_SCHEDULER:-1}" == "1" ]]; then
    args+=(--manual_option_scheduler)
fi
if [[ "$SAVE_VIDEO" == "1" ]]; then
    args+=(--save_video)
fi

echo "[run_eval] RUN_ID=$RUN_ID"
echo "[run_eval] CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "[run_eval] EVAL_CHECKPOINT_STEP=$EVAL_CHECKPOINT_STEP"
echo "[run_eval] EVAL_N_TRAJS=$EVAL_N_TRAJS"
UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache} uv run python "${args[@]}"
