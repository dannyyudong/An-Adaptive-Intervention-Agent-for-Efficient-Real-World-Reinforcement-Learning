#!/usr/bin/env bash
set -euo pipefail

: "${CATKIN_SETUP:?Set CATKIN_SETUP to the ROS workspace setup.bash path.}"
: "${FRANKA_ROBOT_IP:?Set FRANKA_ROBOT_IP explicitly.}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$CATKIN_SETUP"

ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11511}
ROS_PORT=${ROS_PORT:-11511}
export ROS_MASTER_URI

exec python "$SCRIPT_DIR/franka_eggflip_server.py"     --robot_ip="$FRANKA_ROBOT_IP"     --gripper_type=None     --flask_url="${FLASK_URL:-127.0.0.2}"     --ros_port="$ROS_PORT"
