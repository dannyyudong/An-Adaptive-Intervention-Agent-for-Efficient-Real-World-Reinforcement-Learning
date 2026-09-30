"""AIA scheduler configuration for the calibrated USB-insertion task."""

from __future__ import annotations

import os

import numpy as np

from experiments.aia.config import AIATrainingConfigMixin
from experiments.usb_insert.config import (
    EnvConfig as BaselineEnvConfig,
    TrainConfig as BaselineTrainConfig,
)


def _optional_xyz_env(name: str) -> np.ndarray:
    """Return a configured xyz target, or NaNs to disable CodePolicy safely."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return np.full(3, np.nan, dtype=np.float32)

    values = raw.replace(",", " ").split()
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates, got {raw!r}")
    xyz = np.asarray([float(value) for value in values], dtype=np.float32)
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"{name} must contain finite coordinates, got {raw!r}")
    return xyz


class EnvConfig(BaselineEnvConfig):
    """Preserve the baseline USB calibration and add AIA CodePolicy settings."""

    CODE_POLICY_USB_TARGET = _optional_xyz_env("CODE_POLICY_USB_TARGET")
    CODE_POLICY_USB_APPROACH_DZ = float(
        os.getenv("CODE_POLICY_USB_APPROACH_DZ", "0.020")
    )
    CODE_POLICY_USB_POSITION_TOL = float(
        os.getenv("CODE_POLICY_USB_POSITION_TOL", "0.002")
    )
    CODE_POLICY_USB_ROTATION_TOL = float(
        os.getenv("CODE_POLICY_USB_ROTATION_TOL", "0.03")
    )
    CODE_POLICY_USB_MOVE_MAX_STEPS = int(
        os.getenv("CODE_POLICY_USB_MOVE_MAX_STEPS", "50")
    )
    CODE_POLICY_USB_INSERT_MAX_STEPS = int(
        os.getenv("CODE_POLICY_USB_INSERT_MAX_STEPS", "40")
    )
    CODE_POLICY_USB_OPTION_MAX_STEPS = int(
        os.getenv("CODE_POLICY_USB_OPTION_MAX_STEPS", "100")
    )
    CODE_POLICY_USB_CONTACT_STOP_FORCE_Z = float(
        os.getenv("CODE_POLICY_USB_CONTACT_STOP_FORCE_Z", "4.5")
    )


class TrainConfig(AIATrainingConfigMixin, BaselineTrainConfig):
    """USB-insertion RLPD using the AIA option scheduler."""

    checkpoint_period = 1000

    scheduler_trajectory_demo_path = os.getenv(
        "SCHEDULER_TRAJECTORY_DEMO_PATH",
        os.getenv("DEMO_PATH", ""),
    )
    scheduler_trajectory_episode_index = int(
        os.getenv("SCHEDULER_TRAJECTORY_EPISODE_INDEX", "0")
    )
    scheduler_trajectory_window_length = int(
        os.getenv("SCHEDULER_TRAJECTORY_WINDOW_LENGTH", "20")
    )
    scheduler_trajectory_trigger_threshold = float(
        os.getenv("SCHEDULER_TRAJECTORY_TRIGGER_THRESHOLD", "0.005")
    )
    scheduler_trajectory_target_threshold = float(
        os.getenv("SCHEDULER_TRAJECTORY_TARGET_THRESHOLD", "0.002")
    )
    scheduler_trajectory_rotation_trigger_threshold = float(
        os.getenv("SCHEDULER_TRAJECTORY_ROTATION_TRIGGER_THRESHOLD", "0.20")
    )
    scheduler_trajectory_rotation_target_threshold = float(
        os.getenv("SCHEDULER_TRAJECTORY_ROTATION_TARGET_THRESHOLD", "0.05")
    )
    scheduler_trajectory_max_connection_distance = float(
        os.getenv("SCHEDULER_TRAJECTORY_MAX_CONNECTION_DISTANCE", "0.080")
    )
    scheduler_trajectory_require_forward_direction = False
    scheduler_trajectory_max_steps = int(
        os.getenv("SCHEDULER_TRAJECTORY_MAX_STEPS", "30")
    )
    scheduler_rl_horizon = int(os.getenv("SCHEDULER_RL_HORIZON", "5"))

    scheduler_rl_probe_enabled = os.getenv("SCHEDULER_RL_PROBE_ENABLED", "1") != "0"
    scheduler_rl_probe_initial_steps = int(
        os.getenv("SCHEDULER_RL_PROBE_INITIAL_STEPS", "5")
    )
    scheduler_rl_probe_step_increment = int(
        os.getenv("SCHEDULER_RL_PROBE_STEP_INCREMENT", "5")
    )
    _scheduler_rl_probe_max_steps = os.getenv("SCHEDULER_RL_PROBE_MAX_STEPS")
    scheduler_rl_probe_max_steps = (
        int(_scheduler_rl_probe_max_steps)
        if _scheduler_rl_probe_max_steps is not None
        else None
    )
    scheduler_rl_probe_required_passes = int(
        os.getenv("SCHEDULER_RL_PROBE_REQUIRED_PASSES", "2")
    )
    scheduler_rl_probe_episode_interval = int(
        os.getenv("SCHEDULER_RL_PROBE_EPISODE_INTERVAL", "2")
    )
    scheduler_rl_probe_min_progress_delta = float(
        os.getenv("SCHEDULER_RL_PROBE_MIN_PROGRESS_DELTA", "0.01")
    )
    scheduler_rl_probe_expert_progress_fraction = float(
        os.getenv("SCHEDULER_RL_PROBE_EXPERT_PROGRESS_FRACTION", "0.8")
    )
    scheduler_rl_probe_reference_motion_floor_m = float(
        os.getenv("SCHEDULER_RL_PROBE_REFERENCE_MOTION_FLOOR_M", "0.001")
    )
    scheduler_rl_probe_stationary_path_tolerance_m = float(
        os.getenv("SCHEDULER_RL_PROBE_STATIONARY_PATH_TOLERANCE_M", "0.005")
    )
    scheduler_rl_probe_max_path_deviation = float(
        os.getenv(
            "SCHEDULER_RL_PROBE_MAX_PATH_DEVIATION",
            str(scheduler_trajectory_max_connection_distance),
        )
    )
    scheduler_rl_probe_stall_steps = int(
        os.getenv("SCHEDULER_RL_PROBE_STALL_STEPS", "10")
    )
    scheduler_rl_probe_progress_epsilon = float(
        os.getenv("SCHEDULER_RL_PROBE_PROGRESS_EPSILON", "0.001")
    )
    scheduler_rl_probe_progress_epsilon_m = float(
        os.getenv("SCHEDULER_RL_PROBE_PROGRESS_EPSILON_M", "0.0001")
    )
    scheduler_rl_probe_lookahead = int(os.getenv("SCHEDULER_RL_PROBE_LOOKAHEAD", "20"))
    scheduler_rl_probe_initial_search_steps = int(
        os.getenv("SCHEDULER_RL_PROBE_INITIAL_SEARCH_STEPS", "1")
    )
    scheduler_rl_probe_skip_leading_stationary = (
        os.getenv("SCHEDULER_RL_PROBE_SKIP_LEADING_STATIONARY", "1") != "0"
    )
    scheduler_rl_probe_max_index_advance = int(
        os.getenv("SCHEDULER_RL_PROBE_MAX_INDEX_ADVANCE", "0")
    )
    scheduler_rl_probe_max_arc_advance_ratio = float(
        os.getenv("SCHEDULER_RL_PROBE_MAX_ARC_ADVANCE_RATIO", "1.5")
    )
    scheduler_rl_probe_arc_advance_slack_m = float(
        os.getenv("SCHEDULER_RL_PROBE_ARC_ADVANCE_SLACK_M", "0.002")
    )
    scheduler_rl_probe_rotation_weight = float(
        os.getenv("SCHEDULER_RL_PROBE_ROTATION_WEIGHT", "0.01")
    )
    scheduler_rl_probe_off_path_decrement = int(
        os.getenv("SCHEDULER_RL_PROBE_OFF_PATH_DECREMENT", "5")
    )
    scheduler_rl_probe_safety_decrement = int(
        os.getenv("SCHEDULER_RL_PROBE_SAFETY_DECREMENT", "10")
    )

    scheduler_demo_buffer_options = (
        "TRAJECTORY_CORRECTION",
        "CODE_POLICY",
    )
    scheduler_dqn_batch_size = 64
    scheduler_dqn_warmup_transitions = scheduler_dqn_batch_size
    scheduler_dqn_exploration_weights = tuple(
        float(weight.strip())
        for weight in os.getenv(
            "SCHEDULER_DQN_EXPLORATION_WEIGHTS",
            "0.3,0.5,0.2",
        ).split(",")
    )

    def get_code_policy_components(self, env):
        """Build the USB-specific components consumed by CodePolicyOption."""
        from .code_policy import USBInsertionPlanProvider, usb_stage_reached

        raw_env = env.unwrapped

        def current_pose():
            update_pose = getattr(raw_env, "_update_currpos", None)
            if callable(update_pose):
                update_pose()
            pose = np.asarray(
                getattr(raw_env, "curr_pos", []), dtype=np.float32
            ).reshape(-1)
            if pose.shape != (7,):
                raise RuntimeError(
                    "USB CodePolicy requires env.unwrapped.curr_pos with shape (7,)"
                )
            return pose.copy()

        provider = USBInsertionPlanProvider(
            target_xyz=EnvConfig.CODE_POLICY_USB_TARGET,
            current_pose_fn=current_pose,
            approach_dz=EnvConfig.CODE_POLICY_USB_APPROACH_DZ,
            position_tolerance=EnvConfig.CODE_POLICY_USB_POSITION_TOL,
            rotation_tolerance=EnvConfig.CODE_POLICY_USB_ROTATION_TOL,
            move_max_steps=EnvConfig.CODE_POLICY_USB_MOVE_MAX_STEPS,
            insert_max_steps=EnvConfig.CODE_POLICY_USB_INSERT_MAX_STEPS,
            workspace_low=EnvConfig.ABS_POSE_LIMIT_LOW,
            workspace_high=EnvConfig.ABS_POSE_LIMIT_HIGH,
            contact_stop_force_z=EnvConfig.CODE_POLICY_USB_CONTACT_STOP_FORCE_Z,
        )
        return {
            "plan_provider": provider,
            "stage_reached_fn": lambda obs, stage: usb_stage_reached(env, stage),
            "max_steps": EnvConfig.CODE_POLICY_USB_OPTION_MAX_STEPS,
        }
