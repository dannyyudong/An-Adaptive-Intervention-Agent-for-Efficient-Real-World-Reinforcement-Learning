#!/usr/bin/env bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
EXP_DIR=$SCRIPT_DIR
source "$SCRIPT_DIR/../ablation_env.sh"
cd "$EXP_DIR"

: "${DEMO_PATH:?Set DEMO_PATH=/abs/path/to/usb_insert_demo.pkl before starting the learner.}"

RUN_ID_FILE=${RUN_ID_FILE:-$EXP_DIR/.latest_run_id}
RESUME_REQUESTED=${RESUME_TRAINING:-0}
for arg in "$@"; do
    if [[ "$arg" == "--resume_training" || "$arg" == "--resume_training=true" ]]; then
        RESUME_REQUESTED=1
    fi
done

if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
    if [[ -n "${RUN_ID:-}" ]]; then
        :
    elif [[ "$RESUME_REQUESTED" == "1" && -f "$RUN_ID_FILE" ]]; then
        RUN_ID=$(<"$RUN_ID_FILE")
    else
        RUN_ID=$(date +%Y%m%d_%H%M%S)
        mkdir -p "$(dirname "$RUN_ID_FILE")"
        printf '%s\n' "$RUN_ID" > "$RUN_ID_FILE"
    fi
    CHECKPOINT_PATH=$EXP_DIR/aia_usb_insert_${RUN_ID}
fi

RESUME_FLAG=()
if [[ "$RESUME_REQUESTED" == "1" ]]; then
    if [[ ! -e "$CHECKPOINT_PATH" ]]; then
        echo "Cannot resume: CHECKPOINT_PATH does not exist: $CHECKPOINT_PATH" >&2
        exit 1
    fi
    case " $* " in
        *" --resume_training"*|*" --resume_training=true"*) ;;
        *) RESUME_FLAG+=(--resume_training) ;;
    esac
elif [[ -e "$CHECKPOINT_PATH" ]]; then
    echo "Refusing to auto-resume existing CHECKPOINT_PATH: $CHECKPOINT_PATH" >&2
    echo "Set RESUME_TRAINING=1 or pass --resume_training to continue it." >&2
    exit 1
fi

mkdir -p "$(dirname "$CHECKPOINT_PATH")"
printf '%s\n' "$(basename "$CHECKPOINT_PATH" | sed 's/^aia_usb_insert_//')" > "$RUN_ID_FILE"

SCHEDULER_FLAG=()
if [[ "${LEARNED_OPTION_SCHEDULER:-0}" == "1" ]]; then
    SCHEDULER_FLAG+=(--learned_option_scheduler)
elif [[ "${MANUAL_OPTION_SCHEDULER:-1}" == "1" ]]; then
    SCHEDULER_FLAG+=(--manual_option_scheduler)
fi

echo "[run_learner] checkpoint_path=$CHECKPOINT_PATH"
echo "[run_learner] scheduler_flags=${SCHEDULER_FLAG[*]:-disabled}"

python "$ROOT_DIR/examples/train_aia.py" \
    "$@" "${RESUME_FLAG[@]}" "${SCHEDULER_FLAG[@]}" \
    --exp_name=aia_usb_insert \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --demo_path="$DEMO_PATH" \
    --learner
