#!/usr/bin/env bash
set -euo pipefail

# Launch the AIA USB-insertion learner and actor in one tmux session.
# Set NEW_RUN=1 for a fresh run or RESUME_TRAINING=1 to restore an existing run.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
EXP_DIR=$SCRIPT_DIR
source "$SCRIPT_DIR/../ablation_env.sh"
RUN_ID_FILE=${RUN_ID_FILE:-$EXP_DIR/.latest_run_id}
NEW_RUN=${NEW_RUN:-0}
START_LEARNER=${START_LEARNER:-1}
START_ACTOR=${START_ACTOR:-1}
ATTACH_ONLY=${ATTACH_ONLY:-0}
ACTOR_SCRIPT=${ACTOR_SCRIPT:-run_actor.sh}

cd "$EXP_DIR"

if [[ "$NEW_RUN" == "1" && "${RESUME_TRAINING:-0}" == "1" ]]; then
    echo "NEW_RUN=1 and RESUME_TRAINING=1 conflict." >&2
    exit 1
fi

if [[ "$NEW_RUN" == "1" ]]; then
    RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
    unset RESUME_TRAINING
    unset CHECKPOINT_PATH
elif [[ -n "${RUN_ID:-}" ]]; then
    :
elif [[ -f "$RUN_ID_FILE" ]]; then
    RUN_ID=$(<"$RUN_ID_FILE")
else
    RUN_ID=$(date +%Y%m%d_%H%M%S)
fi

export RUN_ID
export CHECKPOINT_PATH=${CHECKPOINT_PATH:-$EXP_DIR/aia_usb_insert_${RUN_ID}}

export EXP_NAME=aia_usb_insert
export WANDB_DESCRIPTOR_SUFFIX=${WANDB_DESCRIPTOR_SUFFIX:-}

DEFAULT_VENV=${VENV:-$ROOT_DIR/.venv}
NVIDIA_LD_LIBRARY_PATH=$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cudnn/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cublas/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cusparse/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cusolver/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/nvjitlink/lib
export VENV=$DEFAULT_VENV
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH=$NVIDIA_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$NVIDIA_LD_LIBRARY_PATH
fi

printf '%s\n' "$RUN_ID" > "$RUN_ID_FILE"

SESSION=${TMUX_SESSION:-${TMUX_SESSION_PREFIX:-aia_usb_insert}_${RUN_ID}}
LEARNER_LOG=$EXP_DIR/tmux_learner_${RUN_ID}.log
ACTOR_LOG=$EXP_DIR/tmux_actor_${RUN_ID}.log

if [[ "$START_LEARNER" == "1" && -z "${DEMO_PATH:-}" ]]; then
    echo "DEMO_PATH is required when START_LEARNER=1." >&2
    echo "It may point to the baseline usb_insert demo_data." >&2
    exit 1
fi

if [[ "$START_ACTOR" == "1" ]]; then
    for required_name in UR_ROBOT_IP HAND_SERIAL EXTERNAL_SERIAL; do
        if [[ -z "${!required_name:-}" ]]; then
            echo "Set $required_name explicitly when START_ACTOR=1." >&2
            exit 1
        fi
    done
    if [[ ( "${LEARNED_OPTION_SCHEDULER:-0}" == "1" || "${MANUAL_OPTION_SCHEDULER:-1}" == "1" ) \
          && -z "${CODE_POLICY_USB_TARGET:-}" ]]; then
        echo "Set CODE_POLICY_USB_TARGET='x y z' after calibrating the USB target." >&2
        exit 1
    fi
fi

if [[ ! -x "$EXP_DIR/$ACTOR_SCRIPT" ]]; then
    echo "Actor script is missing or not executable: $EXP_DIR/$ACTOR_SCRIPT" >&2
    exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed or not on PATH." >&2
    exit 1
fi

shell_quote() {
    printf "%q" "$1"
}

write_env_if_set_or_unset() {
    local target=$1
    local name=$2
    if [[ -n "${!name+x}" ]]; then
        printf "export %s=%s\n" "$name" "$(shell_quote "${!name}")" >> "$target"
    else
        printf "unset %s\n" "$name" >> "$target"
    fi
}

write_common_env() {
    local target=$1
    {
        echo "#!/usr/bin/env bash"
        echo "set -euo pipefail"
        printf "export AIA_ABLATION=%s\n" "$(shell_quote "${AIA_ABLATION:-}")"
        if [[ -n "${AIA_SEED+x}" ]]; then
            printf "export AIA_SEED=%s\n" "$(shell_quote "$AIA_SEED")"
        else
            echo "unset AIA_SEED"
        fi
        if [[ -n "${SCHEDULER_DQN_EXPLORATION_WEIGHTS+x}" ]]; then
            printf "export SCHEDULER_DQN_EXPLORATION_WEIGHTS=%s\n" "$(shell_quote "$SCHEDULER_DQN_EXPLORATION_WEIGHTS")"
        else
            echo "unset SCHEDULER_DQN_EXPLORATION_WEIGHTS"
        fi
        printf "cd %s\n" "$(shell_quote "$EXP_DIR")"
        printf "export RUN_ID=%s\n" "$(shell_quote "$RUN_ID")"
        printf "export CHECKPOINT_PATH=%s\n" "$(shell_quote "$CHECKPOINT_PATH")"
        printf "export EXP_NAME=%s\n" "$(shell_quote "$EXP_NAME")"
        printf "export WANDB_DESCRIPTOR_SUFFIX=%s\n" "$(shell_quote "$WANDB_DESCRIPTOR_SUFFIX")"
        printf "export VENV=%s\n" "$(shell_quote "$VENV")"
        printf "export LD_LIBRARY_PATH=%s\n" "$(shell_quote "$LD_LIBRARY_PATH")"
        printf "export UR_ROBOT_IP=%s\n" "$(shell_quote "${UR_ROBOT_IP:-}")"
        printf "export DOWNWARD_FORCE_BACKOFF_N=%s\n" "$(shell_quote "${DOWNWARD_FORCE_BACKOFF_N:-3.5}")"
        printf "export HAND_SERIAL=%s\n" "$(shell_quote "${HAND_SERIAL:-}")"
        printf "export HAND_EXPOSURE=%s\n" "$(shell_quote "${HAND_EXPOSURE:-15000}")"
        printf "export EXTERNAL_SERIAL=%s\n" "$(shell_quote "${EXTERNAL_SERIAL:-}")"
        printf "export EXTERNAL_WIDTH=%s\n" "$(shell_quote "${EXTERNAL_WIDTH:-640}")"
        printf "export EXTERNAL_HEIGHT=%s\n" "$(shell_quote "${EXTERNAL_HEIGHT:-480}")"
        printf "export CAPTURE_FPS=%s\n" "$(shell_quote "${CAPTURE_FPS:-15}")"
        printf "export EXTERNAL_EXPOSURE=%s\n" "$(shell_quote "${EXTERNAL_EXPOSURE:-18000}")"
        printf "export UV_CACHE_DIR=%s\n" "$(shell_quote "${UV_CACHE_DIR:-/tmp/uv-cache}")"
        printf "export LEARNED_OPTION_SCHEDULER=%s\n" "$(shell_quote "${LEARNED_OPTION_SCHEDULER:-0}")"
        printf "export MANUAL_OPTION_SCHEDULER=%s\n" "$(shell_quote "${MANUAL_OPTION_SCHEDULER:-1}")"
        printf "export SCHEDULER_TRAJECTORY_DEMO_PATH=%s\n" \
            "$(shell_quote "${SCHEDULER_TRAJECTORY_DEMO_PATH:-${DEMO_PATH:-}}")"
        printf "export CODE_POLICY_USB_TARGET=%s\n" \
            "$(shell_quote "${CODE_POLICY_USB_TARGET:-}")"
        printf "export CODE_POLICY_USB_APPROACH_DZ=%s\n" \
            "$(shell_quote "${CODE_POLICY_USB_APPROACH_DZ:-0.020}")"
    } > "$target"

    for optional_var in \
        SCHEDULER_TRAJECTORY_EPISODE_INDEX \
        SCHEDULER_TRAJECTORY_WINDOW_LENGTH \
        SCHEDULER_TRAJECTORY_TRIGGER_THRESHOLD \
        SCHEDULER_TRAJECTORY_TARGET_THRESHOLD \
        SCHEDULER_TRAJECTORY_ROTATION_TRIGGER_THRESHOLD \
        SCHEDULER_TRAJECTORY_ROTATION_TARGET_THRESHOLD \
        SCHEDULER_TRAJECTORY_MAX_CONNECTION_DISTANCE \
        SCHEDULER_TRAJECTORY_MAX_STEPS \
        SCHEDULER_DQN_EXPLORATION_WEIGHTS \
        SCHEDULER_RL_HORIZON \
        SCHEDULER_RL_PROBE_ENABLED \
        SCHEDULER_RL_PROBE_INITIAL_STEPS \
        SCHEDULER_RL_PROBE_STEP_INCREMENT \
        SCHEDULER_RL_PROBE_MAX_STEPS \
        SCHEDULER_RL_PROBE_REQUIRED_PASSES \
        SCHEDULER_RL_PROBE_EPISODE_INTERVAL \
        SCHEDULER_RL_PROBE_MIN_PROGRESS_DELTA \
        SCHEDULER_RL_PROBE_EXPERT_PROGRESS_FRACTION \
        SCHEDULER_RL_PROBE_REFERENCE_MOTION_FLOOR_M \
        SCHEDULER_RL_PROBE_STATIONARY_PATH_TOLERANCE_M \
        SCHEDULER_RL_PROBE_MAX_PATH_DEVIATION \
        SCHEDULER_RL_PROBE_STALL_STEPS \
        SCHEDULER_RL_PROBE_PROGRESS_EPSILON \
        SCHEDULER_RL_PROBE_PROGRESS_EPSILON_M \
        SCHEDULER_RL_PROBE_LOOKAHEAD \
        SCHEDULER_RL_PROBE_INITIAL_SEARCH_STEPS \
        SCHEDULER_RL_PROBE_SKIP_LEADING_STATIONARY \
        SCHEDULER_RL_PROBE_MAX_INDEX_ADVANCE \
        SCHEDULER_RL_PROBE_MAX_ARC_ADVANCE_RATIO \
        SCHEDULER_RL_PROBE_ARC_ADVANCE_SLACK_M \
        SCHEDULER_RL_PROBE_ROTATION_WEIGHT \
        SCHEDULER_RL_PROBE_OFF_PATH_DECREMENT \
        SCHEDULER_RL_PROBE_SAFETY_DECREMENT \
        CODE_POLICY_USB_POSITION_TOL \
        CODE_POLICY_USB_ROTATION_TOL \
        CODE_POLICY_USB_MOVE_MAX_STEPS \
        CODE_POLICY_USB_INSERT_MAX_STEPS \
        CODE_POLICY_USB_OPTION_MAX_STEPS \
        CODE_POLICY_USB_CONTACT_STOP_FORCE_Z
    do
        write_env_if_set_or_unset "$target" "$optional_var"
    done
}

append_actor_rtde_cleanup() {
    local target=$1
    cat >> "$target" <<'SH'

rtde_stop_robot() {
    local robot_ip=${UR_ROBOT_IP:-${ROBOT_IP:-}}
    if [[ -z "$robot_ip" ]]; then
        return 0
    fi

    local py=python
    if [[ -n "${VENV:-}" && -x "$VENV/bin/python" ]]; then
        py="$VENV/bin/python"
    fi

    "$py" - "$robot_ip" <<'PYRTDE'
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
    print(f"[actor] RTDE stop sent to {robot_ip}", flush=True)
except Exception as exc:
    print(f"[actor] RTDE stop skipped: {exc}", flush=True)
PYRTDE
}

cleanup_actor() {
    local status=$?
    trap - EXIT INT TERM HUP
    rtde_stop_robot
    exit "$status"
}
trap cleanup_actor EXIT INT TERM HUP
SH
}

TMUX_CMD_DIR=${TMUX_CMD_DIR:-/tmp/aia_usb_insert_tmux_${RUN_ID}}
LEARNER_CMD_FILE=$TMUX_CMD_DIR/learner.sh
ACTOR_CMD_FILE=$TMUX_CMD_DIR/actor.sh
mkdir -p "$TMUX_CMD_DIR"

write_common_env "$LEARNER_CMD_FILE"
{
    printf "export DEMO_PATH=%s\n" "$(shell_quote "${DEMO_PATH:-}")"
    if [[ "${RESUME_TRAINING:-0}" == "1" ]]; then
        echo "export RESUME_TRAINING=1"
    else
        echo "unset RESUME_TRAINING"
    fi
    echo "echo '[learner] RUN_ID='$RUN_ID"
    echo "echo '[learner] CHECKPOINT_PATH='$CHECKPOINT_PATH"
    echo "echo '[learner] DEMO_PATH='\$DEMO_PATH"
    echo "echo '[learner] SCHEDULER_TRAJECTORY_DEMO_PATH='\$SCHEDULER_TRAJECTORY_DEMO_PATH"
    printf "UV_CACHE_DIR=\$UV_CACHE_DIR uv run bash run_learner.sh 2>&1 | tee -a %s\n" \
        "$(shell_quote "$LEARNER_LOG")"
} >> "$LEARNER_CMD_FILE"

write_common_env "$ACTOR_CMD_FILE"
append_actor_rtde_cleanup "$ACTOR_CMD_FILE"
{
    if [[ "${RESUME_TRAINING_ACTOR:-0}" == "1" ]]; then
        echo "export RESUME_TRAINING=1"
    else
        echo "unset RESUME_TRAINING"
    fi
    echo "echo '[actor] RUN_ID='$RUN_ID"
    echo "echo '[actor] CHECKPOINT_PATH='$CHECKPOINT_PATH"
    echo "echo '[actor] CODE_POLICY_USB_TARGET='\$CODE_POLICY_USB_TARGET"
    printf "UV_CACHE_DIR=\$UV_CACHE_DIR uv run bash %s 2>&1 | tee -a %s\n" \
        "$(shell_quote "$ACTOR_SCRIPT")" "$(shell_quote "$ACTOR_LOG")"
} >> "$ACTOR_CMD_FILE"
chmod +x "$LEARNER_CMD_FILE" "$ACTOR_CMD_FILE"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION"
    exec tmux attach -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" -n train
tmux set-option -t "$SESSION" mouse on
if [[ "$START_LEARNER" == "1" ]]; then
    tmux send-keys -t "$SESSION:train.0" "bash $(shell_quote "$LEARNER_CMD_FILE")" C-m
else
    tmux send-keys -t "$SESSION:train.0" "cd $EXP_DIR; echo 'learner disabled: START_LEARNER=0'; zsh" C-m
fi

tmux split-window -h -t "$SESSION:train.0"
if [[ "$START_ACTOR" == "1" ]]; then
    tmux send-keys -t "$SESSION:train.1" "bash $(shell_quote "$ACTOR_CMD_FILE")" C-m
else
    tmux send-keys -t "$SESSION:train.1" "cd $EXP_DIR; echo 'actor disabled: START_ACTOR=0'; zsh" C-m
fi

tmux select-pane -t "$SESSION:train.0" -T learner
tmux select-pane -t "$SESSION:train.1" -T actor
tmux select-layout -t "$SESSION:train" even-horizontal

echo "Started tmux session: $SESSION"
echo "RUN_ID=$RUN_ID"
echo "EXP_NAME=$EXP_NAME"
echo "CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "ACTOR_SCRIPT=$ACTOR_SCRIPT"
echo "Learner log: $LEARNER_LOG"
echo "Actor log: $ACTOR_LOG"
exec tmux attach -t "$SESSION"
