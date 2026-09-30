import os

import numpy as np

from experiments.config import DefaultTrainingConfig
from experiments.usb_insert.wrapper import (
    GripperPenaltyWrapper,
    UR7ERAMEnv,
)
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from ur_env.envs.ur5_env import DefaultEnvConfig
from ur_env.envs.wrappers import (
    GripperCloseEnv,
    Quat2MrpWrapper,
    SpacemouseIntervention,
)


class UR7EUSBEnv(UR7ERAMEnv):
    """USB insertion task using the RAM insertion contact/reset mechanics."""


class EnvConfig(DefaultEnvConfig):
    ROBOT_IP = os.getenv("UR_ROBOT_IP", os.getenv("ROBOT_IP", ""))
    CONTROLLER_HZ = 200

    # Start from the calibrated RAM insertion pose. Tune these on the USB setup
    # if the fixture/workpiece position differs.
    HOME_TCP_POSE = np.array(
        [
            0.20486,
            -0.35849,
            0.30598,
            -2.19859,
            2.24117,
            -0.04159,
        ],  # -0.02743, -0.39683, 0.30519
        dtype=np.float32,
    )
    RESET_Q = np.array(
        [[0.9519, -1.7670, 1.9762, -1.7274, -1.5715, -0.5454]],
        dtype=np.float32,
    )
    RESET_HEIGHT = 0.12
    RESET_LIFT_Z = float(os.getenv("RESET_LIFT_Z", "0.03"))
    RESET_TIMEOUT_S = float(os.getenv("RESET_TIMEOUT_S", "8.0"))
    RESET_POS_TOL = float(os.getenv("RESET_POS_TOL", "0.002"))
    RESET_ROT_TOL = float(os.getenv("RESET_ROT_TOL", "0.03"))
    RESET_WAIT_DT = float(os.getenv("RESET_WAIT_DT", "0.005"))
    RESET_SPEED_MULTIPLIER = float(os.getenv("RESET_SPEED_MULTIPLIER", "3.0"))
    RESET_FORCE_LIMIT_SCALE = float(os.getenv("RESET_FORCE_LIMIT_SCALE", "3.0"))
    RESET_TARGET_Z_MAX = float(os.getenv("RESET_TARGET_Z_MAX", "0.35"))
    RESET_MAX_TRANSLATION = float(os.getenv("RESET_MAX_TRANSLATION", "0.16"))
    RESET_MAX_ROT_DELTA = float(os.getenv("RESET_MAX_ROT_DELTA", "0.35"))
    RESET_NOISE_ENABLED = os.getenv("RESET_NOISE_ENABLED", "0") != "0"
    RESET_XYZ_NOISE = np.array(
        [
            float(os.getenv("RESET_NOISE_X", "0.003")),
            float(os.getenv("RESET_NOISE_Y", "0.003")),
            float(os.getenv("RESET_NOISE_Z", "0.001")),
        ],
        dtype=np.float32,
    )
    MOVE_HOME_ON_CLOSE = False
    RELEASE_GRIPPER_ON_CLOSE = False
    Z_MAX = float(os.getenv("USB_Z_MAX", os.getenv("RAM_Z_MAX", "0.40")))

    ABS_POSE_LIMIT_LOW = np.array(
        [
            HOME_TCP_POSE[0] - 0.04,
            HOME_TCP_POSE[1] - 0.03,
            0.08,
            -2.19859,
            -2.24117,
            -0.05,
        ],
        dtype=np.float32,
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            HOME_TCP_POSE[0] + 0.04,
            HOME_TCP_POSE[1] + 0.03,
            min(float(HOME_TCP_POSE[2] + 0.01), Z_MAX),
            2.19859,
            2.24117,
            0.05,
        ],
        dtype=np.float32,
    )
    ABS_POSE_RANGE_LIMITS = np.array([-0.02, 0.02], dtype=np.float32)

    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_ROT_RANGE = (0.0,)

    ACTION_SCALE = np.array([0.05, 0, 1.0], dtype=np.float32)
    EXECUTED_ACTION_MASK = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    POLICY_ACTION_MASK = EXECUTED_ACTION_MASK
    SPACEMOUSE_INVERT_AXES = np.array([-1, 1, -1, -1, 1, -1], dtype=np.float32)
    TOOL_ROLL_ONLY = False
    TOOL_ROLL_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    MAX_EPISODE_LENGTH = 200

    CONTROLLER_KP = 1200.0
    CONTROLLER_KD = 120.0
    CONTROLLER_ROT_KP = 35.0
    CONTROLLER_ROT_KD = 4.0
    ERROR_DELTA = 0.02
    DOWNWARD_FORCE_BACKOFF_N = float(os.getenv("DOWNWARD_FORCE_BACKOFF_N", "3.5"))
    TRUNCATE_FORCE_N = 80.0
    FORCEMODE_DAMPING = 0.08
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array(
        [
            float(os.getenv("FORCEMODE_XY_SPEED", "0.55")),
            float(os.getenv("FORCEMODE_XY_SPEED", "0.55")),
            float(os.getenv("FORCEMODE_Z_SPEED", "0.45")),
            float(os.getenv("FORCEMODE_ROT_SPEED", "1.00")),
            float(os.getenv("FORCEMODE_ROT_SPEED", "1.00")),
            float(os.getenv("FORCEMODE_ROT_SPEED", "1.00")),
        ],
        dtype=np.float32,
    )

    GRIPPER_ENABLED = True
    GRIPPER_COMMUNICATION = "socket"
    GRIPPER_PORT = int(os.getenv("GRIPPER_PORT", "63352"))
    GRIPPER_SPEED = 90
    GRIPPER_FORCE = 10
    GRIPPER_TIMEOUT = 500
    GRIPPER_AUTO_CALIBRATE = False
    RESET_GRIPPER_ACTION = os.getenv("RESET_GRIPPER_ACTION", "close")
    RESET_GRIPPER_SETTLE_S = float(os.getenv("RESET_GRIPPER_SETTLE_S", "0.15"))
    SPACEMOUSE_GRIPPER_ENABLED = True

    REALSENSE_CAMERAS = {
        "wrist": {
            "serial_number": os.getenv("HAND_SERIAL", ""),
            "dim": (
                int(os.getenv("HAND_WIDTH", "640")),
                int(os.getenv("HAND_HEIGHT", "480")),
            ),
            "fps": int(os.getenv("CAPTURE_FPS", "15")),
        },
        "external": {
            "serial_number": os.getenv("EXTERNAL_SERIAL", ""),
            "dim": (
                int(os.getenv("EXTERNAL_WIDTH", "640")),
                int(os.getenv("EXTERNAL_HEIGHT", "480")),
            ),
            "fps": int(os.getenv("CAPTURE_FPS", "15")),
            # "exposure": int(os.getenv("EXTERNAL_EXPOSURE", "15000")),
            "exposure": int(os.getenv("EXTERNAL_EXPOSURE", "12000")),
        },
    }
    IMAGE_CROP = {
        "wrist": [0, 480, 0, 640],
        # "external": [30, 403, 22, 406],
        # "external": [137, 474, 109, 496],
        "external": [0, 480, 0, 640],
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist", "external"]
    classifier_keys = ["wrist", "external"]
    proprio_keys = [
        "tcp_pose",
        "tcp_force",
    ]

    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"
    max_traj_length = EnvConfig.MAX_EPISODE_LENGTH
    buffer_period = 2000
    checkpoint_period = 2000
    steps_per_update = 50

    use_reward_classifier = False
    classifier_ckpt_path = os.path.abspath("classifier_ckpt/")

    # Frozen RL representation + four completed-Option motion summaries.
    scheduler_expected_rl_feature_dim = 576
    scheduler_history_length = 4
    scheduler_max_option_duration = EnvConfig.MAX_EPISODE_LENGTH
    scheduler_history_reward_scale = 1.0
    scheduler_history_position_scale_m = 0.05
    scheduler_history_rotation_scale_rad = 0.2
    scheduler_history_path_length_scale_m = 0.10
    scheduler_history_force_scale_n = EnvConfig.TRUNCATE_FORCE_N

    # Option-level reward: task return minus autonomous-assistance and
    # normalized-duration costs. These are conservative initial values and
    # should be calibrated from collected Scheduler transitions.
    scheduler_trajectory_cost = 0.02
    scheduler_code_policy_cost = 0.02
    scheduler_duration_cost = 0.1
    scheduler_replay_buffer_capacity = 50_000

    # Task-level overrides for SchedulerDQNConfig defaults.
    scheduler_dqn_hidden_dims = (256, 256)
    scheduler_dqn_learning_rate = 3e-4
    scheduler_dqn_batch_size = 64
    scheduler_dqn_warmup_transitions = 256
    scheduler_dqn_updates_per_transition = 1
    scheduler_dqn_max_updates_per_loop = 16
    scheduler_dqn_recent_sample_fraction = 0.80
    scheduler_dqn_recent_sample_window = 5_000
    scheduler_dqn_target_update_tau = 0.005
    scheduler_dqn_gradient_clip = 10.0
    scheduler_dqn_epsilon_start = 0.30
    scheduler_dqn_epsilon_end = 0.05
    scheduler_dqn_epsilon_decay_steps = 5_000

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = UR7EUSBEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
            max_episode_length=EnvConfig.MAX_EPISODE_LENGTH,
            hz=10,
            camera_mode="rgb",
        )

        if not fake_env:
            env = SpacemouseIntervention(env)

        if self.setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
            env = GripperCloseEnv(env, fixed_gripper_action=0.0)

        env = Quat2MrpWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env
