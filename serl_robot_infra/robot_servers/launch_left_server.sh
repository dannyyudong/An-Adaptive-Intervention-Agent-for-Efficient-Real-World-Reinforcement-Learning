#!/usr/bin/env bash
set -euo pipefail

: "${CATKIN_SETUP:?Set CATKIN_SETUP to the ROS workspace setup.bash path.}"
: "${FRANKA_ROBOT_IP:?Set FRANKA_ROBOT_IP explicitly.}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$CATKIN_SETUP"

ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}
ROS_PORT=${ROS_PORT:-11311}
GRIPPER_TYPE=${GRIPPER_TYPE:-Franka}
export ROS_MASTER_URI

args=(
    --robot_ip="$FRANKA_ROBOT_IP"
    --gripper_type="$GRIPPER_TYPE"
    --reset_joint_target="${RESET_JOINT_TARGET:-0,0,0,-1.9,-0,2,0}"
    --flask_url="${FLASK_URL:-127.0.0.1}"
    --ros_port="$ROS_PORT"
)
if [[ "$GRIPPER_TYPE" == "Robotiq" ]]; then
    : "${ROBOTIQ_GRIPPER_IP:?Set ROBOTIQ_GRIPPER_IP for a Robotiq gripper.}"
    args+=(--gripper_ip="$ROBOTIQ_GRIPPER_IP")
fi

exec python "$SCRIPT_DIR/franka_server.py" "${args[@]}"
