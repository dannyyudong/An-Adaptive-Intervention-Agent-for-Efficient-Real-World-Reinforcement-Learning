#!/usr/bin/env bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
EXP_DIR=$SCRIPT_DIR
source "$SCRIPT_DIR/../ablation_env.sh"
cd "$EXP_DIR"

for required_name in UR_ROBOT_IP HAND_SERIAL EXTERNAL_SERIAL; do
    if [[ -z "${!required_name:-}" ]]; then
        echo "Set $required_name explicitly before starting the actor." >&2
        exit 1
    fi
done

if [[ ( "${LEARNED_OPTION_SCHEDULER:-0}" == "1" || "${MANUAL_OPTION_SCHEDULER:-1}" == "1" ) \
      && -z "${CODE_POLICY_USB_TARGET:-}" ]]; then
    echo "Set CODE_POLICY_USB_TARGET='x y z' after calibrating the USB target." >&2
    exit 1
fi

rtde_stop_robot() {
    local robot_ip=${UR_ROBOT_IP:-${ROBOT_IP:-}}
    if [[ -z "$robot_ip" ]]; then
        return 0
    fi

    python - "$robot_ip" <<'PYRTDE'
import sys

robot_ip = sys.argv[1]
try:
    from rtde_control import RTDEControlInterface

    control = RTDEControlInterface(robot_ip)
    for stop in (
        lambda: control.forceModeStop(),
        lambda: control.servoStop(),
        lambda: control.speedStop(a=2.0),
        lambda: control.stopScript(),
    ):
        try:
            stop()
        except Exception:
            pass
    control.disconnect()
    print(f"[run_actor] RTDE stop sent to {robot_ip}", flush=True)
except Exception as exc:
    print(f"[run_actor] RTDE stop skipped: {exc}", flush=True)
PYRTDE
}

cleanup_actor() {
    local status=$?
    trap - EXIT INT TERM HUP
    rtde_stop_robot
    exit "$status"
}
trap cleanup_actor EXIT INT TERM HUP

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
    elif [[ -f "$RUN_ID_FILE" ]]; then
        RUN_ID=$(<"$RUN_ID_FILE")
    else
        RUN_ID=$(date +%Y%m%d_%H%M%S)
        mkdir -p "$(dirname "$RUN_ID_FILE")"
        printf '%s\n' "$RUN_ID" > "$RUN_ID_FILE"
    fi
    CHECKPOINT_PATH=$EXP_DIR/aia_usb_insert_${RUN_ID}
fi

RESUME_FLAG=()
ATTACH_FLAG=()
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
    ATTACH_FLAG+=(--allow_existing_checkpoint_path)
    echo "[run_actor] attaching without restoring: $CHECKPOINT_PATH"
fi

SCHEDULER_FLAG=()
if [[ "${LEARNED_OPTION_SCHEDULER:-0}" == "1" ]]; then
    SCHEDULER_FLAG+=(--learned_option_scheduler)
elif [[ "${MANUAL_OPTION_SCHEDULER:-1}" == "1" ]]; then
    SCHEDULER_FLAG+=(--manual_option_scheduler)
fi

echo "[run_actor] checkpoint_path=$CHECKPOINT_PATH"
echo "[run_actor] scheduler_flags=${SCHEDULER_FLAG[*]:-disabled}"

python "$ROOT_DIR/examples/train_aia.py" \
    "$@" "${RESUME_FLAG[@]}" "${ATTACH_FLAG[@]}" "${SCHEDULER_FLAG[@]}" \
    --exp_name=aia_usb_insert \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --actor
