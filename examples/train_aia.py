#!/usr/bin/env python3

import glob
from experiments.aia.ablation import apply_ablation, check_manifest, probabilities
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
import threading
from collections.abc import Mapping
import select
import sys
import termios
import tty
from scipy.spatial.transform import Rotation as R
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.wrappers.chunking import stack_obs

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import (
    MemoryEfficientReplayBufferDataStore,
    SchedulerReplayBufferDataStore,
)
from serl_launcher.aia import (
    AdaptiveRLProbeController,
    AutoSERLTrajectoryConnector,
    CodePolicyOption,
    ExpertTrajectoryProgressEstimator,
    FrozenObservationEncoder,
    OPTION_HISTORY_ITEM_DIM,
    POLICY_CHANGE_PROBE_OUTCOME_FEATURE_DIM,
    POLICY_CHANGE_PROBE_SCHEDULER_STATE_SCHEMA_VERSION,
    OptionID,
    OptionHistory,
    OptionHistoryConfig,
    OptionMotionAccumulator,
    PolicyChangeProbeConfig,
    PolicyChangeProbeController,
    PolicySignature,
    RLOption,
    RLProbeConfig,
    SchedulerDQNAgent,
    SchedulerDQNConfig,
    SchedulerRewardConfig,
    SchedulerStateBuilder,
    SchedulerTransitionBuilder,
    TerminationReason,
    TrajectoryCorrectionOption,
    extract_serl_tcp_pose,
    load_demo_tcp_pose_episodes,
    resolve_leading_motion_start_index,
)
from serl_launcher.aia.trajectory_correct import (
    trajectory_correction_gripper_action,
)

from experiments.mappings import CONFIG_MAPPING
import math

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name", "aia_usb_insert", "Name of experiment corresponding to folder."
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_boolean(
    "resume_training",
    False,
    "Resume from an existing checkpoint_path. Defaults to starting a new run.",
)
flags.DEFINE_boolean(
    "allow_existing_checkpoint_path",
    False,
    "Allow actor to attach to an active run directory without restoring from disk.",
)
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean(
    "manual_option_scheduler",
    False,
    "Enable keyboard selection of RL, trajectory correction, and CodePolicy options.",
)
flags.DEFINE_boolean(
    "learned_option_scheduler",
    False,
    "Use masked epsilon-greedy Double DQN scheduling from the first Option boundary.",
)

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def resolve_rl_probe_max_steps(
    expert_pose_count,
    step_quantum,
    configured_max_steps=None,
):
    """Resolve the full expert-path probe budget unless explicitly overridden."""

    expert_pose_count = int(expert_pose_count)
    step_quantum = int(step_quantum)
    if expert_pose_count <= 0:
        raise ValueError("expert_pose_count must be positive")
    if step_quantum <= 0:
        raise ValueError("step_quantum must be positive")
    if configured_max_steps is not None:
        return int(configured_max_steps)

    expert_transition_steps = max(1, expert_pose_count - 1)
    return ((expert_transition_steps + step_quantum - 1) // step_quantum) * step_quantum


def policy_change_probe_enabled(config):
    """Return whether this run enables the learned-Scheduler Probe."""

    return bool(
        FLAGS.learned_option_scheduler
        and getattr(config, "scheduler_policy_change_probe_enabled", False)
    )


def policy_change_probe_state_feature_dim(config):
    """Compute the task-configured evidence dimension without loading demos."""

    if not policy_change_probe_enabled(config):
        return 0
    region_count = int(getattr(config, "scheduler_policy_change_probe_region_count", 0))
    history_length = int(
        getattr(
            config,
            "scheduler_policy_change_probe_evidence_history_length",
            4,
        )
    )
    if region_count <= 0 or history_length <= 0:
        raise ValueError(
            "policy-change Probe requires positive region_count and "
            "evidence_history_length"
        )
    return region_count + 6 + history_length * POLICY_CHANGE_PROBE_OUTCOME_FEATURE_DIM


def make_policy_change_probe_config(config, anchors):
    rl_option_horizon = int(getattr(config, "scheduler_rl_horizon", 5))
    return PolicyChangeProbeConfig(
        use_policy_drift=bool(
            getattr(config, "scheduler_policy_change_probe_use_policy_drift", True)
        ),
        region_names=tuple(anchors.region_names),
        drift_threshold=float(
            getattr(
                config,
                "scheduler_policy_change_probe_drift_threshold",
                0.05,
            )
        ),
        max_age_steps=int(
            getattr(config, "scheduler_policy_change_probe_max_age_steps", 500)
        ),
        budget_window_steps=int(
            getattr(
                config,
                "scheduler_policy_change_probe_budget_window_steps",
                1000,
            )
        ),
        budget_steps=int(
            getattr(config, "scheduler_policy_change_probe_budget_steps", 50)
        ),
        rl_option_horizon=rl_option_horizon,
        region_max_horizons=anchors.region_max_horizons(rl_option_horizon),
        initial_horizon_steps=int(
            getattr(
                config,
                "scheduler_policy_change_probe_initial_horizon_steps",
                rl_option_horizon,
            )
        ),
        horizon_increment_steps=int(
            getattr(
                config,
                "scheduler_policy_change_probe_horizon_increment_steps",
                rl_option_horizon,
            )
        ),
        required_passes=int(
            getattr(config, "scheduler_policy_change_probe_required_passes", 2)
        ),
        off_path_decrement_steps=int(
            getattr(
                config,
                "scheduler_policy_change_probe_off_path_decrement_steps",
                rl_option_horizon,
            )
        ),
        safety_decrement_steps=int(
            getattr(
                config,
                "scheduler_policy_change_probe_safety_decrement_steps",
                2 * rl_option_horizon,
            )
        ),
        max_path_deviation_m=float(
            getattr(
                config,
                "scheduler_policy_change_probe_max_path_deviation_m",
                0.08,
            )
        ),
        evidence_history_length=int(
            getattr(
                config,
                "scheduler_policy_change_probe_evidence_history_length",
                4,
            )
        ),
        task_return_scale=float(
            getattr(
                config,
                "scheduler_policy_change_probe_task_return_scale",
                1.0,
            )
        ),
        discrete_temperature=float(
            getattr(
                config,
                "scheduler_policy_change_probe_discrete_temperature",
                1.0,
            )
        ),
        discrete_weight=float(
            getattr(
                config,
                "scheduler_policy_change_probe_discrete_weight",
                1.0,
            )
        ),
    )


def extract_policy_signature(agent, anchor_observations, continuous_action_mask):
    """Evaluate one actor snapshot on fixed anchors without sampling actions."""

    anchors = tuple(anchor_observations)
    if not anchors:
        raise ValueError("policy-change Probe region has no anchor observations")
    batched_observations = jax.device_put(stack_obs(anchors))
    distribution = agent.forward_policy(batched_observations, train=False)
    base_distribution = getattr(distribution, "distribution", None)
    if base_distribution is None:
        raise TypeError("policy-change Probe requires a Gaussian policy distribution")
    continuous_mean = np.asarray(
        jax.device_get(base_distribution.mean()), dtype=np.float32
    )
    continuous_std = np.asarray(
        jax.device_get(base_distribution.stddev()), dtype=np.float32
    )
    if continuous_mean.ndim != 2 or continuous_std.shape != continuous_mean.shape:
        raise ValueError(
            "policy-change Probe expected batched diagonal Gaussian outputs, got "
            f"{continuous_mean.shape} and {continuous_std.shape}"
        )

    action_mask = np.asarray(continuous_action_mask, dtype=np.float32).reshape(-1)
    if action_mask.size < continuous_mean.shape[-1]:
        raise ValueError(
            "policy-change Probe action mask is shorter than the continuous policy"
        )
    active_dimensions = np.flatnonzero(action_mask[: continuous_mean.shape[-1]] > 0.5)
    if active_dimensions.size == 0:
        raise ValueError(
            "policy-change Probe action mask disables every continuous action"
        )
    continuous_mean = continuous_mean[:, active_dimensions]
    continuous_std = continuous_std[:, active_dimensions]

    discrete_logits = None
    grasp_forward = getattr(agent, "forward_grasp_critic", None)
    if callable(grasp_forward):
        logits = grasp_forward(
            batched_observations,
            rng=jax.random.PRNGKey(0),
            train=False,
        )
        discrete_logits = np.asarray(jax.device_get(logits), dtype=np.float32)
    return PolicySignature(
        continuous_mean=continuous_mean,
        continuous_std=continuous_std,
        discrete_logits=discrete_logits,
    )


def _intervention_flags(transition):
    """Return mutually exclusive ``(human, other_strategy)`` flags."""
    if not isinstance(transition, dict):
        return False, False
    infos = transition.get("infos", {})
    if not isinstance(infos, dict):
        infos = {}

    human = bool(
        transition.get("human_intervention", infos.get("human_intervention", False))
    )
    other_strategy = bool(
        transition.get(
            "other_strategy_intervention",
            infos.get(
                "other_strategy_intervention",
                infos.get("strategy_intervention", False),
            ),
        )
    )
    intervention_source = str(infos.get("intervention_source", "")).lower()
    action_source = str(infos.get("action_source", "")).lower()
    option_name = str(infos.get("option_name", "")).upper()

    if intervention_source in {"human", "spacemouse"} or action_source == "spacemouse":
        human = True
    if (
        intervention_source == "autoserl"
        or intervention_source.startswith("schedule")
        or action_source in {"autoserl", "schedule"}
        or bool(infos.get("primitive_intervention", False))
        or (option_name and option_name != "RL")
    ):
        other_strategy = True

    # A SpaceMouse override owns the executed action and must not be double-counted.
    if human:
        return True, False
    if other_strategy:
        return False, True

    # Backward compatibility for existing SpaceMouse transition files.
    legacy_intervention = bool(
        transition.get("intervention", infos.get("intervention", False))
    )
    return legacy_intervention, False


class InterventionTrackingReplayBufferDataStore(MemoryEfficientReplayBufferDataStore):
    """Replay buffer that tracks interventions by source."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._intervention_stats_lock = threading.Lock()
        self._replay_total_samples = 0
        self._replay_human_intervention_samples = 0
        self._replay_other_strategy_intervention_samples = 0

    def insert(self, *args, **kwargs):
        transition = args[0] if args else kwargs.get("data_dict")
        super().insert(*args, **kwargs)
        if transition is not None:
            human_intervention, other_strategy_intervention = _intervention_flags(
                transition
            )
            with self._intervention_stats_lock:
                self._replay_total_samples += 1
                self._replay_human_intervention_samples += int(human_intervention)
                self._replay_other_strategy_intervention_samples += int(
                    other_strategy_intervention
                )

    def get_intervention_stats(self):
        with self._intervention_stats_lock:
            total_samples = self._replay_total_samples
            human_samples = self._replay_human_intervention_samples
            other_strategy_samples = self._replay_other_strategy_intervention_samples
        intervention_samples = human_samples + other_strategy_samples
        human_ratio = human_samples / total_samples if total_samples else 0.0
        other_strategy_ratio = (
            other_strategy_samples / total_samples if total_samples else 0.0
        )
        intervention_ratio = (
            intervention_samples / total_samples if total_samples else 0.0
        )
        return {
            "total_samples": total_samples,
            "human_intervention_samples": human_samples,
            "human_intervention_ratio": human_ratio,
            "other_strategy_intervention_samples": other_strategy_samples,
            "other_strategy_intervention_ratio": other_strategy_ratio,
            "intervention_samples": intervention_samples,
            "intervention_ratio": intervention_ratio,
        }


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


class ConsecutiveCodePolicyEpisodeGuard:
    """Temporarily mask trajectory correction after repeated CodePolicy episodes."""

    def __init__(self, required_streak=2, trajectory_block_steps=20):
        if int(required_streak) <= 0:
            raise ValueError("required_streak must be positive")
        if int(trajectory_block_steps) < 0:
            raise ValueError("trajectory_block_steps must be non-negative")
        self.required_streak = int(required_streak)
        self.trajectory_block_steps = int(trajectory_block_steps)
        self.consecutive_code_policy_episodes = 0
        self.current_episode_used_code_policy = False
        self.episode_steps = 0
        self.block_trajectory_this_episode = False

    @property
    def trajectory_blocked(self):
        return bool(
            self.block_trajectory_this_episode
            and self.episode_steps < self.trajectory_block_steps
        )

    def mark_option_started(self, option_id):
        if OptionID(option_id) is OptionID.CODE_POLICY:
            self.current_episode_used_code_policy = True

    def record_env_step(self, count=1):
        count = int(count)
        if count < 0:
            raise ValueError("count must be non-negative")
        self.episode_steps += count

    def apply_action_mask(self, action_mask):
        masked = np.asarray(action_mask, dtype=bool).copy()
        if masked.shape != (len(OptionID),):
            raise ValueError(
                f"Scheduler action mask must have shape {(len(OptionID),)}, "
                f"got {masked.shape}"
            )
        if self.trajectory_blocked:
            masked[int(OptionID.TRAJECTORY_CORRECTION)] = False
        return masked

    def finish_episode(self):
        completed_episode_used_code_policy = bool(self.current_episode_used_code_policy)
        if completed_episode_used_code_policy:
            self.consecutive_code_policy_episodes += 1
        else:
            self.consecutive_code_policy_episodes = 0

        self.block_trajectory_this_episode = bool(
            self.consecutive_code_policy_episodes >= self.required_streak
            and self.trajectory_block_steps > 0
        )
        self.current_episode_used_code_policy = False
        self.episode_steps = 0
        return {
            "completed_episode_used_code_policy": completed_episode_used_code_policy,
            "consecutive_code_policy_episodes": (self.consecutive_code_policy_episodes),
            "next_episode_trajectory_block_steps": (
                self.trajectory_block_steps if self.block_trajectory_this_episode else 0
            ),
        }


class ManualEpisodeLabeler:
    def __init__(self):
        self.success = False
        self.failure = False
        self.primitive_intervention = False
        self.option_request = None
        self.scheduler_toggle_requested = False
        self.listener = None
        self.keyboard = None
        self.stdin_fd = None
        self.stdin_settings = None

    def start(self):
        try:
            if sys.stdin.isatty():
                self.stdin_fd = sys.stdin.fileno()
                self.stdin_settings = termios.tcgetattr(self.stdin_fd)
                tty.setcbreak(self.stdin_fd)
        except Exception as exc:
            print(f"Manual stdin labels disabled: {exc}")

        try:
            from pynput import keyboard as pynput_keyboard

            self.keyboard = pynput_keyboard
            self.listener = self.keyboard.Listener(on_press=self._on_press)
            self.listener.start()
        except Exception as exc:
            print(f"Manual pynput labels disabled: {exc}")

        message = (
            "Manual actor labels: SPACE/s = success and reset, "
            "f/ESC = failure and reset, i = primitive intervention."
        )
        if FLAGS.manual_option_scheduler or FLAGS.learned_option_scheduler:
            message += (
                " Option keys: 0 = scheduler on/off, 1 = RL, "
                "2 = trajectory correction, 3 = CodePolicy."
            )
        print(message)

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
        if self.stdin_fd is not None and self.stdin_settings is not None:
            try:
                termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, self.stdin_settings)
            except Exception:
                pass

    def consume(self):
        self._poll_stdin()
        if self.success:
            self.success = False
            self.failure = False
            return "success"
        if self.failure:
            self.failure = False
            return "failure"
        if self.primitive_intervention:
            self.primitive_intervention = False
            return "primitive_intervention"
        return None

    def consume_option_request(self):
        self._poll_stdin()
        option_request = self.option_request
        self.option_request = None
        return option_request

    def consume_scheduler_toggle(self):
        self._poll_stdin()
        toggle_requested = self.scheduler_toggle_requested
        self.scheduler_toggle_requested = False
        return toggle_requested

    def clear(self):
        self._poll_stdin()
        self.success = False
        self.failure = False
        self.primitive_intervention = False
        self.option_request = None

    def _mark_char(self, char):
        if char in (" ", "s", "S"):
            self.success = True
        elif char in ("f", "F", "\x1b"):
            self.failure = True
        elif char in ("i", "I"):
            self.primitive_intervention = True
        elif char == "0":
            self.scheduler_toggle_requested = True
        elif char == "1":
            self.option_request = OptionID.RL
        elif char == "2":
            self.option_request = OptionID.TRAJECTORY_CORRECTION
        elif char == "3":
            self.option_request = OptionID.CODE_POLICY

    def _poll_stdin(self):
        if self.stdin_fd is None:
            return
        try:
            while select.select([sys.stdin], [], [], 0)[0]:
                self._mark_char(sys.stdin.read(1))
        except Exception:
            pass

    def _on_press(self, key):
        if self.keyboard is not None and key == self.keyboard.Key.space:
            self.success = True
            return
        if self.keyboard is not None and key == self.keyboard.Key.esc:
            self.failure = True
            return
        try:
            self._mark_char(key.char)
        except AttributeError:
            pass


def _iter_env_chain(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "env", None)


def _call_env_chain(env, method_name):
    for current in _iter_env_chain(env):
        method = getattr(current, method_name, None)
        if callable(method):
            method()


def _call_env_chain_result(env, method_name, *args, **kwargs):
    for current in _iter_env_chain(env):
        method = getattr(current, method_name, None)
        if callable(method):
            return method(*args, **kwargs)
    raise AttributeError(f"Environment chain has no callable method {method_name!r}")


def _wrap_raw_observation_from_env_chain(env, obs):
    """Apply outer observation wrappers after calling a method on the unwrapped env."""
    wrapped_obs = obs
    chain = list(_iter_env_chain(env))
    for current in reversed(chain[:-1]):
        observation = current.__class__.__dict__.get("observation")
        if callable(observation):
            wrapped_obs = observation(current, wrapped_obs)

        if "current_obs" in current.__dict__ and "obs_horizon" in current.__dict__:
            current.current_obs.clear()
            current.current_obs.extend([wrapped_obs] * current.obs_horizon)
            wrapped_obs = stack_obs(current.current_obs)

        if "last_closed_norm" in current.__dict__:
            try:
                current.last_closed_norm = float(current.unwrapped.gripper_state[0])
            except Exception:
                pass

    return wrapped_obs


def prepare_manual_reset(env, action_filter):
    action_filter.reset()
    _call_env_chain(env, "clear_intervention")


def align_transition_action(transition, action_space):
    if "actions" not in transition:
        return transition

    target_shape = tuple(action_space.shape)
    target_dim = int(np.prod(target_shape))
    action = np.asarray(transition["actions"], dtype=np.float32).reshape(-1)

    if action.shape[0] > target_dim:
        action = action[:target_dim]
    elif action.shape[0] < target_dim:
        action = np.pad(action, (0, target_dim - action.shape[0]), constant_values=0.0)

    transition["actions"] = action.reshape(target_shape).astype(np.float32)
    return transition


def align_observation_state(obs, observation_space):
    if not isinstance(obs, dict) or "state" not in obs:
        return obs
    if (
        not hasattr(observation_space, "spaces")
        or "state" not in observation_space.spaces
    ):
        return obs

    target_shape = tuple(observation_space.spaces["state"].shape)
    target_dim = int(np.prod(target_shape))
    state = np.asarray(obs["state"], dtype=np.float32)
    flat_state = state.reshape(-1)

    if flat_state.shape[0] > target_dim:
        flat_state = flat_state[:target_dim]
    elif flat_state.shape[0] < target_dim:
        flat_state = np.pad(
            flat_state, (0, target_dim - flat_state.shape[0]), constant_values=0.0
        )

    obs = copy.deepcopy(obs)
    obs["state"] = flat_state.reshape(target_shape).astype(np.float32)
    return obs


def align_transition_observations(transition, observation_space):
    for key in ("observations", "next_observations"):
        if key in transition:
            transition[key] = align_observation_state(
                transition[key], observation_space
            )
    return transition


def clip_action_to_space(action, action_space):
    action = np.asarray(action, dtype=np.float32)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)

    if action.shape == low.shape:
        return np.clip(action, low, high).astype(np.float32)

    flat_action = action.reshape(-1)
    flat_low = low.reshape(-1)
    flat_high = high.reshape(-1)
    if flat_action.shape == flat_low.shape:
        return (
            np.clip(flat_action, flat_low, flat_high)
            .reshape(action.shape)
            .astype(np.float32)
        )

    return (
        np.clip(flat_action, np.min(flat_low), np.max(flat_high))
        .reshape(action.shape)
        .astype(np.float32)
    )


def _get_env_attr(env, attr_name, default=None):
    for current in _iter_env_chain(env):
        if attr_name in getattr(current, "__dict__", {}):
            return getattr(current, attr_name)
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, attr_name):
        return getattr(unwrapped, attr_name)
    return default


def _get_env_xyz_action_scale(env):
    scale = _get_env_attr(env, "action_scale", None)
    if scale is None:
        return 0.05
    scale = np.asarray(scale, dtype=np.float32).reshape(-1)
    if scale.size == 0:
        return 0.05
    return max(float(scale[0]), 1e-6)


def _get_env_rot_action_scale(env):
    scale = _get_env_attr(env, "action_scale", None)
    if scale is None:
        return 0.1
    scale = np.asarray(scale, dtype=np.float32).reshape(-1)
    if scale.size < 2:
        return 0.1
    return max(float(scale[1]), 1e-6)


def _get_env_curr_pose(env):
    unwrapped = getattr(env, "unwrapped", env)
    update_currpos = getattr(unwrapped, "_update_currpos", None)
    if callable(update_currpos):
        update_currpos()
    curr_pos = getattr(unwrapped, "curr_pos", None)
    if curr_pos is None:
        raise RuntimeError(
            "primitive intervention env.step plan requires env.unwrapped.curr_pos"
        )
    curr_pos = np.asarray(curr_pos, dtype=np.float32).reshape(-1)
    if curr_pos.size < 7:
        raise RuntimeError(
            f"primitive intervention curr_pos has invalid shape: {curr_pos.shape}"
        )
    return curr_pos[:7].copy()


def _get_env_curr_xyz(env):
    curr_pos = _get_env_curr_pose(env)
    return curr_pos[:3].copy()


def _primitive_pose_action(env, target_xyz=None, target_quat=None, gripper_action=0.0):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    flat_action = action.reshape(-1)
    curr_pose = None
    if target_xyz is not None:
        target_xyz = np.asarray(target_xyz, dtype=np.float32).reshape(3)
        curr_pose = _get_env_curr_pose(env)
        curr_xyz = curr_pose[:3]
        xyz_gain = 2.0
        flat_action[:3] = np.clip(
            xyz_gain * (target_xyz - curr_xyz) / _get_env_xyz_action_scale(env),
            -1.0,
            1.0,
        )
    if target_quat is not None and flat_action.size >= 6:
        if curr_pose is None:
            curr_pose = _get_env_curr_pose(env)
        target_quat = np.asarray(target_quat, dtype=np.float32).reshape(4)
        rot_delta = R.from_quat(target_quat) * R.from_quat(curr_pose[3:7]).inv()
        flat_action[3:6] = np.clip(
            rot_delta.as_mrp() * 4.0 / _get_env_rot_action_scale(env),
            -1.0,
            1.0,
        )
    if flat_action.size >= 7:
        flat_action[6] = float(gripper_action)
    return clip_action_to_space(action, env.action_space)


def _primitive_plan_phases(env, plan):
    envstep = plan.get("envstep", {}) or {}
    safety = plan.get("safety", {}) or {}
    waypoints = plan.get("waypoints", {}) or {}
    max_steps = max(1, int(envstep.get("max_steps_per_waypoint", 80)))
    gripper_steps = max(1, int(envstep.get("gripper_steps", 10)))

    phases = []
    min_start_z = float(safety.get("min_start_z", 0.0))
    if min_start_z > 0.0:
        current_xyz = _get_env_curr_xyz(env)
        if current_xyz[2] < min_start_z:
            lift_xyz = current_xyz.copy()
            lift_xyz[2] = min_start_z
            phases.append(
                {
                    "kind": "move",
                    "name": "start_lift",
                    "target_xyz": lift_xyz,
                    "gripper_action": 0.0,
                    "max_steps": max_steps,
                }
            )

    threshold = float(safety.get("open_gripper_threshold", -1.0))
    closed_norm = float(safety.get("closed_norm", float("nan")))
    if threshold >= 0.0 and (not np.isfinite(closed_norm) or closed_norm > threshold):
        phases.append(
            {
                "kind": "gripper",
                "name": "pre_open",
                "gripper_action": -1.0,
                "steps": gripper_steps,
            }
        )

    phases.extend(
        [
            {
                "kind": "move",
                "name": "pick_approach",
                "target_xyz": waypoints["pick_approach"],
                "gripper_action": 0.0,
                "max_steps": max_steps,
            },
            {
                "kind": "move",
                "name": "pick",
                "target_xyz": waypoints["pick"],
                "gripper_action": 0.0,
                "max_steps": max_steps,
            },
            {
                "kind": "gripper",
                "name": "close",
                "gripper_action": 1.0,
                "steps": gripper_steps,
            },
            {
                "kind": "move",
                "name": "lift_after_pick",
                "target_xyz": waypoints["pick_approach"],
                "gripper_action": 0.0,
                "max_steps": max_steps,
            },
            {
                "kind": "move",
                "name": "place_approach",
                "target_xyz": waypoints["place_approach"],
                "gripper_action": 0.0,
                "max_steps": max_steps,
            },
            {
                "kind": "move",
                "name": "place",
                "target_xyz": waypoints["place"],
                "gripper_action": 0.0,
                "max_steps": max_steps,
            },
            {
                "kind": "gripper",
                "name": "open",
                "gripper_action": -1.0,
                "steps": gripper_steps,
            },
        ]
    )
    return phases


def prepare_loaded_transition(transition, env, include_grasp_penalty):
    transition = copy.deepcopy(transition)
    transition = align_transition_action(transition, env.action_space)
    transition = align_transition_observations(transition, env.observation_space)
    if "infos" in transition and "grasp_penalty" in transition["infos"]:
        transition["grasp_penalty"] = transition["infos"]["grasp_penalty"]
    elif include_grasp_penalty and "grasp_penalty" not in transition:
        transition["grasp_penalty"] = 0.0
    return transition


def latest_step_from_path(path, prefix):
    if not path:
        return None
    name = os.path.basename(path.rstrip(os.sep))
    name = os.path.splitext(name)[0]
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    digits = []
    for ch in suffix:
        if not ch.isdigit():
            break
        digits.append(ch)
    if not digits:
        return None
    return int("".join(digits))


def get_resume_step(checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0

    resume_steps = []

    buffer_files = natsorted(
        glob.glob(os.path.join(checkpoint_path, "buffer", "transitions_*.pkl"))
    )
    for buffer_file in buffer_files:
        step = latest_step_from_path(buffer_file, "transitions_")
        if step is not None:
            resume_steps.append(step + 1)

    latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(checkpoint_path))
    step = latest_step_from_path(latest_ckpt, "checkpoint_")
    if step is not None:
        resume_steps.append(step + 1)

    return max(resume_steps, default=0)


def split_complete_trajectories(transitions):
    last_done = -1
    for idx, transition in enumerate(transitions):
        if bool(transition.get("dones", False)):
            last_done = idx

    if last_done < 0:
        return [], transitions
    return transitions[: last_done + 1], transitions[last_done + 1 :]


def atomic_pickle_dump(obj, path):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_pickle_file(path):
    try:
        with open(path, "rb") as f:
            obj = pkl.load(f)
    except EOFError:
        with open(path, "rb") as f:
            data = f.read()
        obj = pkl.loads(data + b"ue.")
        print(f"Recovered truncated pickle while loading {path}")
    return obj


def load_transition_file(path):
    obj = load_pickle_file(path)
    if isinstance(obj, dict) and {
        "observations",
        "actions",
        "next_observations",
        "rewards",
        "masks",
        "dones",
    }.issubset(obj.keys()):
        return [obj]
    if isinstance(obj, list):
        return [
            x
            for x in obj
            if isinstance(x, dict)
            and {
                "observations",
                "actions",
                "next_observations",
                "rewards",
                "masks",
                "dones",
            }.issubset(x.keys())
        ]
    return []


def load_scheduler_transition_file(path):
    """Load Scheduler transitions, including files saved before masks existed."""
    obj = load_pickle_file(path)
    candidates = (
        [obj] if isinstance(obj, dict) else obj if isinstance(obj, list) else []
    )
    required = {
        "observations",
        "actions",
        "next_observations",
        "rewards",
        "dones",
        "durations",
        "discounts",
    }
    transitions = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not required.issubset(candidate):
            continue
        transition = dict(candidate)
        transition.setdefault(
            "masks", np.float32(1.0 - float(bool(transition["dones"])))
        )
        transition.setdefault("rl_policy_version", np.int64(0))
        transitions.append(transition)
    return transitions


def dump_completed_transitions(transitions, path):
    completed, remainder = split_complete_trajectories(transitions)
    if not completed:
        return remainder, 0
    atomic_pickle_dump(completed, path)
    return remainder, len(completed)


##############################################################################


class EMAActionFilter:
    def __init__(
        self, hz: float, cutoff_hz: float = 2.0, filter_rot=True, filter_gripper=True
    ):
        self.dt = 1.0 / float(hz)
        tau = 1.0 / (2.0 * math.pi * float(cutoff_hz))
        self.alpha = self.dt / (tau + self.dt)
        self.filter_rot = filter_rot
        self.filter_gripper = filter_gripper
        self.prev = None

    def __call__(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32).copy()

        if self.prev is None:
            self.prev = a.copy()
            return a

        # Filter continuous part: up to first 6 dims if present
        idx_end = min(6, a.shape[0])
        self.prev[:idx_end] = self.prev[:idx_end] + self.alpha * (
            a[:idx_end] - self.prev[:idx_end]
        )

        # Only touch gripper if it exists
        if a.shape[0] > 6:
            if self.filter_gripper:
                self.prev[6] = self.prev[6] + self.alpha * (a[6] - self.prev[6])
            else:
                self.prev[6] = a[6]

        return np.clip(self.prev, -1.0, 1.0).astype(np.float32)

    def reset(self):
        self.prev = None


def build_manual_scheduler_options(env, rl_policy_fn):
    """Construct task-configured Options without changing the default actor path."""
    demo_path = getattr(config, "scheduler_trajectory_demo_path", None)
    if not demo_path:
        raise RuntimeError(
            f"Experiment {FLAGS.exp_name!r} does not configure a trajectory demo"
        )
    episodes = load_demo_tcp_pose_episodes(demo_path)
    episode_index = int(getattr(config, "scheduler_trajectory_episode_index", 0))
    if not 0 <= episode_index < len(episodes):
        raise IndexError(
            f"Trajectory episode index {episode_index} is outside [0, {len(episodes)})"
        )

    connector = AutoSERLTrajectoryConnector(
        episodes[episode_index],
        window_length=int(getattr(config, "scheduler_trajectory_window_length", 20)),
        trigger_threshold=float(
            getattr(config, "scheduler_trajectory_trigger_threshold", 0.02)
        ),
        target_threshold=float(
            getattr(config, "scheduler_trajectory_target_threshold", 0.005)
        ),
        rotation_trigger_threshold=getattr(
            config, "scheduler_trajectory_rotation_trigger_threshold", None
        ),
        rotation_target_threshold=getattr(
            config, "scheduler_trajectory_rotation_target_threshold", None
        ),
        max_connection_distance=float(
            getattr(config, "scheduler_trajectory_max_connection_distance", 0.08)
        ),
        require_forward_direction=bool(
            getattr(config, "scheduler_trajectory_require_forward_direction", True)
        ),
    )

    def current_pose(_obs):
        return _get_env_curr_pose(env)

    def trajectory_target(obs):
        return connector.propose(current_pose(obs))

    def trajectory_confidence(obs):
        target = trajectory_target(obs)
        return 0.0 if target is None else target.confidence

    def trajectory_correction_action(obs, target):
        policy_action = np.asarray(rl_policy_fn(obs), dtype=np.float32).reshape(-1)
        action = _primitive_pose_action(
            env,
            target_xyz=target.pose[:3],
            target_quat=target.pose[3:7],
            gripper_action=trajectory_correction_gripper_action(policy_action),
        )
        return action

    trajectory_option = TrajectoryCorrectionOption(
        action_space=env.action_space,
        target_provider=trajectory_target,
        correction_policy_fn=trajectory_correction_action,
        target_reached_fn=lambda obs, target: connector.target_reached(
            current_pose(obs), target
        ),
        max_steps=int(getattr(config, "scheduler_trajectory_max_steps", 20)),
        gamma=float(config.discount),
        availability_fn=lambda obs: trajectory_target(obs) is not None,
        confidence_fn=trajectory_confidence,
        clip_actions=True,
    )

    components_factory = getattr(config, "get_code_policy_components", None)
    if not callable(components_factory):
        raise RuntimeError(
            f"Experiment {FLAGS.exp_name!r} does not provide CodePolicy components"
        )
    components = components_factory(env)

    def code_policy_action(obs, stage):
        current_tcp = _get_env_curr_pose(env)
        action = _primitive_pose_action(
            env,
            target_xyz=stage.target_xyz,
            target_quat=stage.target_quat,
            gripper_action=stage.gripper_action,
        )
        target_parts = []
        if stage.target_xyz is not None:
            target_parts.append(
                np.asarray(stage.target_xyz, dtype=np.float32).reshape(3)
            )
        if stage.target_quat is not None:
            target_parts.append(
                np.asarray(stage.target_quat, dtype=np.float32).reshape(4)
            )
        target_tcp = np.concatenate(target_parts) if target_parts else None
        target_tcp_text = (
            "none" if target_tcp is None else np.array2string(target_tcp, precision=5)
        )
        print(
            "[CodePolicy action] "
            f"stage={stage.name} "
            f"target_tcp={target_tcp_text} "
            f"current_tcp={np.array2string(current_tcp, precision=5)} "
            f"action={np.array2string(action, precision=5)}",
            flush=True,
        )
        return action

    code_policy_option = CodePolicyOption(
        action_space=env.action_space,
        plan_provider=components["plan_provider"],
        primitive_action_fn=code_policy_action,
        stage_reached_fn=components["stage_reached_fn"],
        max_steps=int(components["max_steps"]),
        gamma=float(config.discount),
        clip_actions=True,
    )

    options = {
        OptionID.RL: RLOption(
            policy_fn=rl_policy_fn,
            action_space=env.action_space,
            horizon=int(getattr(config, "scheduler_rl_horizon", 5)),
            gamma=float(config.discount),
            clip_actions=True,
        ),
        OptionID.TRAJECTORY_CORRECTION: trajectory_option,
        OptionID.CODE_POLICY: code_policy_option,
    }
    print(
        "[manual option scheduler] loaded "
        f"episode {episode_index}/{len(episodes) - 1} from {demo_path} "
        f"with {len(episodes[episode_index])} poses",
        flush=True,
    )
    return options, connector


def actor(
    agent,
    scheduler_agent,
    scheduler_dqn_config,
    data_store,
    intvn_data_store,
    scheduler_data_store,
    env,
    sampling_rng,
):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    action_filter = EMAActionFilter(hz=10, cutoff_hz=2)
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        manual_labeler = ManualEpisodeLabeler()
        manual_labeler.start()
        try:
            for episode in range(FLAGS.eval_n_trajs):
                obs, _ = env.reset()
                manual_labeler.clear()
                done = False
                truncated = False
                start_time = time.time()
                while not (done or truncated):
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs), argmax=True, seed=key
                    )
                    actions = np.asarray(jax.device_get(actions))
                    actions = action_filter(actions)
                    next_obs, reward, done, truncated, info = env.step(actions)
                    obs = next_obs

                    manual_label = manual_labeler.consume()
                    if manual_label == "success":
                        reward = 1.0
                        done = True
                        truncated = False
                        print("manual eval success marked; resetting environment")
                    elif manual_label == "failure":
                        reward = 0.0
                        done = True
                        truncated = False
                        print("manual eval failure marked; resetting environment")

                    if done or truncated:
                        if reward:
                            dt = time.time() - start_time
                            time_list.append(dt)
                            print(dt)
                        action_filter.reset()
                        success_counter += reward
                        print(reward)
                        print(f"{success_counter}/{episode + 1}")

            print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
            print(f"average time: {np.mean(time_list) if time_list else float('nan')}")
        finally:
            try:
                manual_labeler.stop()
            except Exception:
                pass
            try:
                env.close()
            except Exception as exc:
                print(f"eval cleanup failed: {exc}")
        return  # after done eval, return and exit

    start_step = (
        get_resume_step(FLAGS.checkpoint_path)
        if (FLAGS.resume_training or FLAGS.allow_existing_checkpoint_path)
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
        "scheduler_env": scheduler_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=500,
    )

    # Policy selection starts immediately from the locally initialized Scheduler.
    # Learner-side data readiness is tracked separately and only gates updates.
    scheduler_policy_ready = bool(FLAGS.learned_option_scheduler)
    scheduler_update_step = int(np.asarray(jax.device_get(scheduler_agent.state.step)))
    rl_policy_version = int(start_step)
    pending_network = {"payload": None, "count": 0, "applied": 0}
    pending_network_lock = threading.Lock()

    def update_params(payload):
        with pending_network_lock:
            pending_network["payload"] = payload
            pending_network["count"] += 1

    def apply_pending_network(force=False):
        nonlocal agent
        nonlocal scheduler_agent
        nonlocal scheduler_policy_ready
        nonlocal scheduler_update_step
        nonlocal rl_policy_version
        if not force and step % config.steps_per_update != 0:
            return False
        with pending_network_lock:
            payload = pending_network["payload"]
            count = pending_network["count"]
            pending_network["payload"] = None
        if payload is not None:
            if isinstance(payload, Mapping) and "rl_params" in payload:
                agent = agent.replace(
                    state=agent.state.replace(params=payload["rl_params"])
                )
                if "scheduler_params" in payload:
                    scheduler_agent = scheduler_agent.replace(
                        state=scheduler_agent.state.replace(
                            params=payload["scheduler_params"]
                        )
                    )
                scheduler_policy_ready = bool(
                    payload.get(
                        "scheduler_policy_ready",
                        payload.get("scheduler_ready", scheduler_policy_ready),
                    )
                )
                scheduler_update_step = int(payload.get("scheduler_step", 0))
                rl_policy_version = int(payload.get("rl_policy_version", 0))
            else:
                # Backward compatibility with learners that publish only RL params.
                agent = agent.replace(state=agent.state.replace(params=payload))
            skipped = max(0, count - pending_network["applied"] - 1)
            pending_network["applied"] = count
            if skipped:
                print_green(
                    "Applied latest RL/Scheduler params; "
                    f"skipped {skipped} stale updates"
                )
            else:
                print_green("Applied latest RL/Scheduler params")
            return True
        return False

    client.recv_network_callback(update_params)

    current_trajectory = []
    current_scheduler_trajectory = []
    code_policy_episode_guard = ConsecutiveCodePolicyEpisodeGuard(
        required_streak=2,
        trajectory_block_steps=20,
    )
    trajectory_index = 0
    scheduler_trajectory_index = 0
    actor_stats = {
        "saved_trajectories": 0,
        "saved_success_trajectories": 0,
        "saved_failure_trajectories": 0,
        "saved_intervention_trajectories": 0,
        "saved_demo_trajectories": 0,
        "saved_policy_trajectories": 0,
        "saved_transitions": 0,
        "saved_intervention_samples": 0,
        "saved_demo_samples": 0,
    }
    policy_change_probe_episode_metrics = {}
    last_buffer_dump_step = start_step
    last_seen_step = start_step
    actor_sync_period = max(1, int(os.getenv("ACTOR_SYNC_PERIOD_STEPS", "10")))
    last_actor_sync_step = start_step

    def mark_transition_intervention(
        transition, intervention, trajectory_had_intervention=False
    ):
        transition["intervention"] = bool(intervention)
        transition["trajectory_had_intervention"] = bool(trajectory_had_intervention)
        info = copy.deepcopy(transition.get("infos", {}))
        info["intervention"] = bool(intervention)
        info["trajectory_had_intervention"] = bool(trajectory_had_intervention)
        source = str(info.get("intervention_source", "")).lower()
        other_strategy = bool(
            info.get("other_strategy_intervention", False)
            or (
                intervention and (source == "autoserl" or source.startswith("schedule"))
            )
        )
        human = bool(
            info.get("human_intervention", False)
            or (intervention and not other_strategy)
        )
        if human:
            other_strategy = False
        info["human_intervention"] = human
        info["other_strategy_intervention"] = other_strategy
        if intervention and not source:
            info["intervention_source"] = "human"
        transition["infos"] = info
        return transition

    def mark_transition_demo_eligibility(transition, eligible, source=None):
        """Tag data for demo replay without changing human-intervention semantics."""
        transition["demo_eligible"] = bool(eligible)
        info = copy.deepcopy(transition.get("infos", {}))
        info["demo_eligible"] = bool(eligible)
        if source is not None:
            source = str(source)
            transition["demo_source"] = source
            info["demo_source"] = source
        transition["infos"] = info
        return transition

    def should_add_terminal_demo_bridge(
        trajectory, reward, terminal, already_demo_eligible=False
    ):
        """Keep an immediate successful terminal transition after demo data."""
        if already_demo_eligible or not terminal or not trajectory:
            return False
        reward_value = float(np.asarray(reward).reshape(()))
        if reward_value <= 0.5:
            return False
        previous = trajectory[-1]
        return bool(
            previous.get("intervention", False) or previous.get("demo_eligible", False)
        )

    def dump_trajectory(trajectory, step):
        nonlocal trajectory_index
        if FLAGS.checkpoint_path is None or not trajectory:
            return 0
        trajectory_had_intvn = any(
            bool(t.get("intervention", False)) for t in trajectory
        )
        reward_value = float(np.asarray(trajectory[-1].get("rewards", 0.0)).reshape(()))
        done_value = bool(trajectory[-1].get("dones", False))
        success = bool(reward_value > 0.5 and done_value)
        label = "success" if success else "failure"
        source = "intvn" if trajectory_had_intvn else "policy"
        trajectory = [
            mark_transition_intervention(
                copy.deepcopy(t), t.get("intervention", False), trajectory_had_intvn
            )
            for t in trajectory
        ]
        filename = f"transitions_{step}_{trajectory_index:06d}_{label}_{source}.pkl"
        buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer", filename)
        atomic_pickle_dump(trajectory, buffer_path)
        intervention_samples = [
            copy.deepcopy(t) for t in trajectory if bool(t.get("intervention", False))
        ]
        demo_samples = [
            copy.deepcopy(t)
            for t in trajectory
            if bool(t.get("intervention", False)) or bool(t.get("demo_eligible", False))
        ]
        if demo_samples:
            demo_filename = f"demo_samples_{step}_{trajectory_index:06d}_{label}.pkl"
            demo_path = os.path.join(
                FLAGS.checkpoint_path, "demo_buffer", demo_filename
            )
            atomic_pickle_dump(demo_samples, demo_path)
        actor_stats["saved_trajectories"] += 1
        actor_stats["saved_success_trajectories"] += int(success)
        actor_stats["saved_failure_trajectories"] += int(not success)
        actor_stats["saved_intervention_trajectories"] += int(trajectory_had_intvn)
        actor_stats["saved_demo_trajectories"] += int(bool(demo_samples))
        actor_stats["saved_policy_trajectories"] += int(not trajectory_had_intvn)
        actor_stats["saved_transitions"] += len(trajectory)
        actor_stats["saved_intervention_samples"] += len(intervention_samples)
        actor_stats["saved_demo_samples"] += len(demo_samples)
        stats_payload = {
            "buffer/actor_saved_trajectories": actor_stats["saved_trajectories"],
            "buffer/actor_saved_success_trajectories": actor_stats[
                "saved_success_trajectories"
            ],
            "buffer/actor_saved_failure_trajectories": actor_stats[
                "saved_failure_trajectories"
            ],
            "buffer/actor_saved_intervention_trajectories": actor_stats[
                "saved_intervention_trajectories"
            ],
            "buffer/actor_saved_demo_trajectories": actor_stats[
                "saved_demo_trajectories"
            ],
            "buffer/actor_saved_policy_trajectories": actor_stats[
                "saved_policy_trajectories"
            ],
            "buffer/actor_saved_transitions": actor_stats["saved_transitions"],
            "buffer/actor_saved_intervention_samples": actor_stats[
                "saved_intervention_samples"
            ],
            "buffer/actor_saved_demo_samples": actor_stats["saved_demo_samples"],
            "buffer/last_trajectory_length": len(trajectory),
            "buffer/last_trajectory_success": int(success),
            "buffer/last_trajectory_intervention": int(trajectory_had_intvn),
            "buffer/last_trajectory_intervention_samples": len(intervention_samples),
            "buffer/last_trajectory_demo_samples": len(demo_samples),
        }
        stats_payload.update(policy_change_probe_episode_metrics)
        try:
            client.request("send-stats", stats_payload)
        except Exception as exc:
            print(f"actor stats log failed: {exc}", flush=True)
        policy_change_probe_episode_metrics.clear()
        trajectory_index += 1
        print(
            f"saved trajectory step={step} len={len(trajectory)} label={label} "
            f"intervention={trajectory_had_intvn}",
            flush=True,
        )
        return len(trajectory)

    def dump_scheduler_trajectory(step, *, final=False):
        nonlocal current_scheduler_trajectory
        nonlocal scheduler_trajectory_index
        if FLAGS.checkpoint_path is None or not current_scheduler_trajectory:
            return 0
        suffix = "partial" if final else "episode"
        filename = (
            f"scheduler_transitions_{step}_{scheduler_trajectory_index:06d}_"
            f"{suffix}.pkl"
        )
        path = os.path.join(FLAGS.checkpoint_path, "scheduler_buffer", filename)
        atomic_pickle_dump(current_scheduler_trajectory, path)
        count = len(current_scheduler_trajectory)
        current_scheduler_trajectory = []
        scheduler_trajectory_index += 1
        print(
            f"saved scheduler transitions step={step} count={count} " f"kind={suffix}",
            flush=True,
        )
        return count

    def dump_actor_buffers(step, final=False):
        nonlocal last_buffer_dump_step
        # Completed trajectories are flushed immediately at episode end.
        last_buffer_dump_step = step

    def sync_actor_data(step, force=False):
        nonlocal last_actor_sync_step
        if not force and step - last_actor_sync_step < actor_sync_period:
            return
        try:
            if client.update():
                last_actor_sync_step = step
        except Exception as exc:
            print(f"actor datastore sync failed: {exc}", flush=True)

    def finish_code_policy_guard_episode():
        if not scheduler_configured:
            return
        summary = code_policy_episode_guard.finish_episode()
        print(
            "[option scheduler] episode guard "
            f"used_code_policy={summary['completed_episode_used_code_policy']} "
            f"consecutive_code_policy_episodes="
            f"{summary['consecutive_code_policy_episodes']} "
            f"next_episode_trajectory_block_steps="
            f"{summary['next_episode_trajectory_block_steps']}",
            flush=True,
        )

    def insert_primitive_envstep_transition(
        obs,
        action,
        next_obs,
        reward,
        done,
        truncated,
        info,
        phase_name,
        step,
        plan,
    ):
        info = copy.deepcopy(info)
        if "left" in info:
            info.pop("left")
        if "right" in info:
            info.pop("right")

        executed_action = info.pop("intervene_action", None)
        spacemouse_override = executed_action is not None
        if not spacemouse_override:
            executed_action = info.pop("executed_action", action)
        else:
            print(
                "[actor primitive intervention] SpaceMouse/action wrapper overrode primitive action",
                flush=True,
            )
        executed_action = clip_action_to_space(executed_action, env.action_space)

        info["primitive_intervention"] = True
        info["primitive_phase"] = phase_name
        info["primitive_name"] = plan.get("primitive", "pick_and_place")
        info["primitive_context"] = plan.get("primitive_context", "intervention")
        info["primitive_placement_mode"] = plan.get("placement_mode")
        info["human_intervention"] = bool(spacemouse_override)
        info["other_strategy_intervention"] = not spacemouse_override
        info["intervention_source"] = (
            "human" if spacemouse_override else "schedule:primitive"
        )

        terminal = bool(done or truncated)
        transition = dict(
            observations=obs,
            actions=executed_action,
            next_observations=next_obs,
            rewards=reward,
            masks=1.0 - float(done),
            dones=terminal,
            infos=info,
        )
        if "grasp_penalty" in info:
            transition["grasp_penalty"] = info["grasp_penalty"]

        transition = mark_transition_intervention(transition, True, True)
        data_store.insert(transition)
        intvn_data_store.insert(copy.deepcopy(transition))
        current_trajectory.append(copy.deepcopy(transition))
        sync_actor_data(step)
        return transition

    def finish_manual_label_episode(obs, manual_label, step):
        nonlocal current_trajectory
        nonlocal running_return
        nonlocal intervention_count
        nonlocal intervention_steps
        nonlocal already_intervened
        nonlocal trajectory_had_intervention

        prepare_manual_reset(env, action_filter)
        reward = 1.0 if manual_label == "success" else 0.0
        info = {
            "succeed": manual_label == "success",
            "episode": {
                "intervention_count": intervention_count,
                "intervention_steps": intervention_steps,
            },
        }
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        transition = dict(
            observations=obs,
            actions=zero_action,
            next_observations=obs,
            rewards=reward,
            masks=0.0,
            dones=True,
            infos=copy.deepcopy(info),
        )
        if config.setup_mode in (
            "single-arm-learned-gripper",
            "dual-arm-learned-gripper",
        ):
            transition["grasp_penalty"] = 0.0
        transition = mark_transition_intervention(
            transition, False, trajectory_had_intervention
        )
        terminal_demo_bridge = should_add_terminal_demo_bridge(
            current_trajectory,
            reward,
            terminal=True,
            already_demo_eligible=False,
        )
        transition = mark_transition_demo_eligibility(
            transition,
            terminal_demo_bridge,
            "terminal_bridge_after_demo" if terminal_demo_bridge else None,
        )
        data_store.insert(transition)
        if terminal_demo_bridge:
            intvn_data_store.insert(copy.deepcopy(transition))
            print(
                "[demo terminal bridge] added pre-step manual success "
                "transition to demo buffer",
                flush=True,
            )
        current_trajectory.append(copy.deepcopy(transition))
        dump_trajectory(current_trajectory, step)
        dump_scheduler_trajectory(step)
        current_trajectory = []
        sync_actor_data(step, force=True)
        running_return += reward
        info["episode"]["intervention_count"] = intervention_count
        info["episode"]["intervention_steps"] = intervention_steps
        print(f"manual {manual_label} marked; resetting environment", flush=True)
        finish_code_policy_guard_episode()
        print("[actor] calling env.reset", flush=True)
        reset_obs, _ = env.reset()
        manual_labeler.clear()
        print("[actor] env.reset returned", flush=True)
        pbar.set_description(f"last return: {running_return}")
        running_return = 0.0
        intervention_count = 0
        intervention_steps = 0
        already_intervened = False
        trajectory_had_intervention = False
        action_filter.reset()
        return reset_obs

    def execute_primitive_plan_through_env_steps(obs, plan, step):
        nonlocal running_return
        nonlocal already_intervened
        nonlocal trajectory_had_intervention
        nonlocal intervention_count
        nonlocal intervention_steps
        nonlocal current_trajectory

        if not bool(plan.get("execute", False)):
            print(
                "[actor primitive intervention] execute=False; planned only, no env.step actions",
                flush=True,
            )
            return obs, False, None

        envstep = plan.get("envstep", {}) or {}
        xyz_tolerance = float(envstep.get("xyz_tolerance", 0.015))
        rot_tolerance = float(envstep.get("rot_tolerance", 0.08))
        target_quat = plan.get("home_quat", None)
        phases = _primitive_plan_phases(env, plan)

        trajectory_had_intervention = True
        intervention_count += 1
        already_intervened = True
        total_steps = 0

        def consume_manual_interrupt(phase_name):
            manual_label = manual_labeler.consume()
            if manual_label in ("success", "failure"):
                print(
                    f"[actor primitive intervention] interrupted by manual {manual_label} "
                    f"during phase={phase_name}",
                    flush=True,
                )
                return manual_label
            if manual_label == "primitive_intervention":
                print(
                    "[actor primitive intervention] ignored nested primitive intervention request",
                    flush=True,
                )
            return None

        for phase in phases:
            phase_name = str(phase["name"])
            print(
                f"[actor primitive intervention] env.step phase={phase_name}",
                flush=True,
            )

            if phase["kind"] == "move":
                target_xyz = np.asarray(phase["target_xyz"], dtype=np.float32).reshape(
                    3
                )
                max_steps = max(1, int(phase.get("max_steps", 80)))
                for phase_step in range(max_steps):
                    try:
                        curr_pose = _get_env_curr_pose(env)
                        pos_err = float(np.linalg.norm(curr_pose[:3] - target_xyz))
                        if target_quat is None:
                            rot_err = 0.0
                        else:
                            rot_err = float(
                                (
                                    R.from_quat(
                                        np.asarray(
                                            target_quat, dtype=np.float32
                                        ).reshape(4)
                                    )
                                    * R.from_quat(curr_pose[3:7]).inv()
                                ).magnitude()
                            )
                    except Exception:
                        pos_err = float("inf")
                        rot_err = float("inf")
                    if pos_err <= xyz_tolerance and rot_err <= rot_tolerance:
                        print(
                            f"[actor primitive intervention] phase={phase_name} reached "
                            f"pos_err={pos_err:.4f} rot_err={rot_err:.4f} steps={phase_step}",
                            flush=True,
                        )
                        break

                    action = _primitive_pose_action(
                        env,
                        target_xyz=target_xyz,
                        target_quat=target_quat,
                        gripper_action=float(phase.get("gripper_action", 0.0)),
                    )
                    next_obs, reward, done, truncated, info = env.step(action)
                    insert_primitive_envstep_transition(
                        obs,
                        action,
                        next_obs,
                        reward,
                        done,
                        truncated,
                        info,
                        phase_name,
                        step,
                        plan,
                    )
                    obs = next_obs
                    code_policy_episode_guard.record_env_step()
                    running_return += reward
                    intervention_steps += 1
                    total_steps += 1
                    if done or truncated:
                        return obs, True, None
                    manual_interrupt = consume_manual_interrupt(phase_name)
                    if manual_interrupt is not None:
                        already_intervened = False
                        return obs, False, manual_interrupt
                else:
                    try:
                        curr_pose = _get_env_curr_pose(env)
                        pos_err = float(np.linalg.norm(curr_pose[:3] - target_xyz))
                        if target_quat is None:
                            rot_err = 0.0
                        else:
                            rot_err = float(
                                (
                                    R.from_quat(
                                        np.asarray(
                                            target_quat, dtype=np.float32
                                        ).reshape(4)
                                    )
                                    * R.from_quat(curr_pose[3:7]).inv()
                                ).magnitude()
                            )
                    except Exception:
                        pos_err = float("inf")
                        rot_err = float("inf")
                    print(
                        f"[actor primitive intervention] phase={phase_name} max_steps reached "
                        f"pos_err={pos_err:.4f} rot_err={rot_err:.4f}",
                        flush=True,
                    )

            elif phase["kind"] == "gripper":
                gripper_steps = max(1, int(phase.get("steps", 1)))
                gripper_action = float(phase.get("gripper_action", 0.0))
                for _ in range(gripper_steps):
                    action = _primitive_pose_action(
                        env,
                        target_xyz=None,
                        target_quat=target_quat,
                        gripper_action=gripper_action,
                    )
                    next_obs, reward, done, truncated, info = env.step(action)
                    insert_primitive_envstep_transition(
                        obs,
                        action,
                        next_obs,
                        reward,
                        done,
                        truncated,
                        info,
                        phase_name,
                        step,
                        plan,
                    )
                    obs = next_obs
                    code_policy_episode_guard.record_env_step()
                    running_return += reward
                    intervention_steps += 1
                    total_steps += 1
                    if done or truncated:
                        return obs, True, None
                    manual_interrupt = consume_manual_interrupt(phase_name)
                    if manual_interrupt is not None:
                        already_intervened = False
                        return obs, False, manual_interrupt

        already_intervened = False
        print(
            f"[actor primitive intervention] env.step plan finished steps={total_steps}",
            flush=True,
        )
        return obs, False, None

    def finish_terminal_episode(step, info):
        nonlocal current_trajectory
        nonlocal running_return
        nonlocal intervention_count
        nonlocal intervention_steps
        nonlocal already_intervened
        nonlocal trajectory_had_intervention

        info.setdefault("episode", {})
        info["episode"]["intervention_count"] = intervention_count
        info["episode"]["intervention_steps"] = intervention_steps
        pbar.set_description(f"last return: {running_return}")
        dump_trajectory(current_trajectory, step)
        dump_scheduler_trajectory(step)
        current_trajectory = []
        sync_actor_data(step, force=True)
        running_return = 0.0
        intervention_count = 0
        intervention_steps = 0
        already_intervened = False
        trajectory_had_intervention = False
        prepare_manual_reset(env, action_filter)
        finish_code_policy_guard_episode()
        print("[actor] calling env.reset", flush=True)
        reset_obs, _ = env.reset()
        manual_labeler.clear()
        print("[actor] env.reset returned", flush=True)
        action_filter.reset()
        return reset_obs

    manual_labeler = ManualEpisodeLabeler()
    manual_labeler.start()

    obs, _ = env.reset()
    manual_labeler.clear()
    action_filter.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    trajectory_had_intervention = False
    intervention_count = 0
    intervention_steps = 0
    primitive_intervention_pending = False
    scheduler_configured = bool(
        FLAGS.manual_option_scheduler or FLAGS.learned_option_scheduler
    )
    scheduler_enabled = scheduler_configured
    fixed_option_scheduler = getattr(config, "aia_ablation", None) == "fixed_rule"
    demo_buffer_option_ids = frozenset(
        OptionID[name] for name in getattr(config, "scheduler_demo_buffer_options", ())
    )
    active_option = None
    active_scheduler_type = "disabled"
    selected_option_id = OptionID.RL
    manual_option_override_pending = False
    manual_options = {}
    trajectory_connector = None
    rl_probe_controller = None
    rl_probe_progress_estimator = None
    rl_probe_state_path = None
    policy_change_probe_controller = None
    policy_change_probe_anchors = None
    policy_change_probe_state_path = None
    policy_change_probe_signature_version = None
    policy_change_probe_action_mask = None
    active_policy_change_probe_decision = None
    active_autonomous_region = None
    active_autonomous_signature = None
    active_base_option_id = None
    option_step_context = {"step": start_step}
    scheduler_history = None
    scheduler_state_builder = None
    scheduler_transition_builder = None
    scheduler_reward_config = None
    scheduler_pending_transition = None
    option_motion_accumulator = None
    scheduler_boundary_cache = {
        "observation": None,
        "state": None,
        "action_mask": None,
    }

    def clear_scheduler_boundary_cache():
        scheduler_boundary_cache["observation"] = None
        scheduler_boundary_cache["state"] = None
        scheduler_boundary_cache["action_mask"] = None

    def rl_option_policy(option_obs):
        nonlocal sampling_rng
        current_step = int(option_step_context["step"])
        if current_step < config.random_steps:
            return env.action_space.sample()
        sampling_rng, key = jax.random.split(sampling_rng)
        action = agent.sample_actions(
            observations=jax.device_put(option_obs),
            seed=key,
            argmax=False,
        )
        return action_filter(np.asarray(jax.device_get(action)))

    def current_policy_change_progress(current_observation=None):
        if policy_change_probe_controller is None:
            return None
        progress = trajectory_connector.latest_progress
        if progress is None:
            if current_observation is None:
                current_pose = _get_env_curr_pose(env)
            else:
                current_pose = extract_serl_tcp_pose(
                    {"observations": current_observation}
                )
            progress = trajectory_connector.observe(current_pose)
        return progress

    def current_policy_change_region(current_observation=None):
        progress = current_policy_change_progress(current_observation)
        if progress is None:
            return None
        return policy_change_probe_anchors.region_for_progress_index(progress.index)

    def refresh_policy_change_probe_signatures(*, force=False):
        nonlocal policy_change_probe_signature_version
        if policy_change_probe_controller is None:
            return False
        if not policy_change_probe_controller.config.use_policy_drift:
            policy_change_probe_controller.current_policy_version = int(
                rl_policy_version
            )
            return False
        if not force and policy_change_probe_signature_version == rl_policy_version:
            return False
        started = time.perf_counter()
        signatures = tuple(
            extract_policy_signature(
                agent,
                region_anchors,
                policy_change_probe_action_mask,
            )
            for region_anchors in policy_change_probe_anchors.anchor_observations
        )
        policy_change_probe_controller.set_current_signatures(
            signatures,
            policy_version=rl_policy_version,
        )
        policy_change_probe_signature_version = int(rl_policy_version)
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        print(
            "[policy-change probe] refreshed fixed-anchor policy signatures "
            f"version={rl_policy_version} elapsed_ms={elapsed_ms:.1f}",
            flush=True,
        )
        return True

    def save_policy_change_probe_state():
        if (
            policy_change_probe_controller is not None
            and policy_change_probe_state_path is not None
        ):
            atomic_pickle_dump(
                policy_change_probe_controller.state_dict(),
                policy_change_probe_state_path,
            )

    if scheduler_configured:
        manual_options, trajectory_connector = build_manual_scheduler_options(
            env, rl_option_policy
        )
        history_config = OptionHistoryConfig(
            length=int(getattr(config, "scheduler_history_length", 4)),
            max_duration=int(
                getattr(config, "scheduler_max_option_duration", config.max_traj_length)
            ),
            reward_scale=float(getattr(config, "scheduler_history_reward_scale", 1.0)),
            position_scale_m=float(
                getattr(config, "scheduler_history_position_scale_m", 0.05)
            ),
            rotation_scale_rad=float(
                getattr(config, "scheduler_history_rotation_scale_rad", 0.2)
            ),
            path_length_scale_m=float(
                getattr(config, "scheduler_history_path_length_scale_m", 0.10)
            ),
            force_scale_n=float(
                getattr(config, "scheduler_history_force_scale_n", 80.0)
            ),
        )
        encoder_warmup_started = time.perf_counter()
        frozen_encoder = FrozenObservationEncoder.from_agent(
            agent,
            obs,
            expected_feature_dim=int(
                getattr(config, "scheduler_expected_rl_feature_dim", 576)
            ),
        )
        encoder_warmup_ms = 1000.0 * (time.perf_counter() - encoder_warmup_started)
        scheduler_history = OptionHistory(history_config)
        use_policy_change_probe = policy_change_probe_enabled(config)
        use_legacy_probe = bool(
            FLAGS.learned_option_scheduler
            and getattr(config, "scheduler_rl_probe_enabled", False)
        )
        if use_policy_change_probe and use_legacy_probe:
            raise ValueError(
                "Policy-change Probe and legacy AdaptiveRLProbe cannot both be "
                "enabled; disable one for a clean experimental condition"
            )
        extra_feature_dim = 0
        extra_feature_fn = None
        state_schema_version = None
        if use_policy_change_probe:
            anchor_provider = getattr(config, "get_policy_change_probe_anchors", None)
            if not callable(anchor_provider):
                raise RuntimeError(
                    "policy-change Probe requires a task-local fixed-anchor provider"
                )
            policy_change_probe_anchors = anchor_provider()
            expected_region_count = int(
                getattr(
                    config,
                    "scheduler_policy_change_probe_region_count",
                    0,
                )
            )
            if policy_change_probe_anchors.region_count != expected_region_count:
                raise ValueError(
                    "policy-change Probe anchor/config region counts differ: "
                    f"{policy_change_probe_anchors.region_count} vs "
                    f"{expected_region_count}"
                )
            policy_change_config = make_policy_change_probe_config(
                config, policy_change_probe_anchors
            )
            policy_change_probe_controller = PolicyChangeProbeController(
                policy_change_config,
                anchor_fingerprint=policy_change_probe_anchors.fingerprint,
            )
            raw_config = getattr(env.unwrapped, "config", None)
            policy_change_probe_action_mask = np.asarray(
                getattr(
                    raw_config,
                    "POLICY_ACTION_MASK",
                    np.ones(env.action_space.shape, dtype=np.float32),
                ),
                dtype=np.float32,
            ).reshape(-1)
            if FLAGS.checkpoint_path is not None:
                policy_change_probe_state_path = os.path.join(
                    os.path.abspath(FLAGS.checkpoint_path),
                    "scheduler_policy_change_probe_state.pkl",
                )
                if (
                    FLAGS.resume_training or FLAGS.allow_existing_checkpoint_path
                ) and os.path.exists(policy_change_probe_state_path):
                    policy_change_probe_controller.restore_state(
                        load_pickle_file(policy_change_probe_state_path)
                    )
                    print_green("Loaded policy-change Probe regional evidence.")
            refresh_policy_change_probe_signatures(force=True)
            extra_feature_dim = policy_change_probe_controller.state_feature_dim
            configured_extra_dim = policy_change_probe_state_feature_dim(config)
            if extra_feature_dim != configured_extra_dim:
                raise ValueError(
                    "policy-change Probe state dimension differs between actor "
                    f"and learner configuration: {extra_feature_dim} vs "
                    f"{configured_extra_dim}"
                )
            extra_feature_fn = lambda state_obs: (
                policy_change_probe_controller.state_features(
                    region=current_policy_change_region(state_obs),
                    step=int(option_step_context["step"]),
                )
            )
            state_schema_version = POLICY_CHANGE_PROBE_SCHEDULER_STATE_SCHEMA_VERSION
        scheduler_state_builder = SchedulerStateBuilder(
            frozen_encoder,
            history_config,
            extra_feature_dim=extra_feature_dim,
            extra_feature_fn=extra_feature_fn,
            **(
                {}
                if state_schema_version is None
                else {"schema_version": state_schema_version}
            ),
        )
        scheduler_reward_config = SchedulerRewardConfig(
            gamma=float(config.discount),
            trajectory_cost=float(getattr(config, "scheduler_trajectory_cost", 0.02)),
            code_policy_cost=float(getattr(config, "scheduler_code_policy_cost", 0.02)),
            duration_cost=float(getattr(config, "scheduler_duration_cost", 0.01)),
            max_duration=history_config.max_duration,
        )
        scheduler_transition_builder = SchedulerTransitionBuilder(
            scheduler_state_builder.state_dim,
            scheduler_reward_config,
            state_schema_version=scheduler_state_builder.schema_version,
        )
        rl_probe_enabled = use_legacy_probe
        if rl_probe_enabled:
            rl_horizon = int(getattr(config, "scheduler_rl_horizon", 5))
            probe_initial_steps = int(
                getattr(config, "scheduler_rl_probe_initial_steps", rl_horizon)
            )
            probe_reference_motion_floor_m = float(
                os.getenv(
                    "SCHEDULER_RL_PROBE_REFERENCE_MOTION_FLOOR_M",
                    str(
                        getattr(
                            config,
                            "scheduler_rl_probe_reference_motion_floor_m",
                            0.001,
                        )
                    ),
                )
            )
            probe_skip_leading_stationary = (
                os.getenv(
                    "SCHEDULER_RL_PROBE_SKIP_LEADING_STATIONARY",
                    "1"
                    if getattr(
                        config,
                        "scheduler_rl_probe_skip_leading_stationary",
                        False,
                    )
                    else "0",
                )
                != "0"
            )
            expert_original_pose_count = len(trajectory_connector.demo_tcp_poses)
            probe_start_index = (
                resolve_leading_motion_start_index(
                    trajectory_connector.demo_tcp_poses,
                    probe_initial_steps,
                    probe_reference_motion_floor_m,
                )
                if probe_skip_leading_stationary
                else 0
            )
            probe_reference_pose_count = expert_original_pose_count - probe_start_index
            configured_probe_max_steps = getattr(
                config, "scheduler_rl_probe_max_steps", None
            )
            probe_max_steps = resolve_rl_probe_max_steps(
                probe_reference_pose_count,
                rl_horizon,
                configured_probe_max_steps,
            )
            probe_max_steps_source = (
                "expert_trajectory"
                if configured_probe_max_steps is None
                else "config_override"
            )
            rl_probe_config = RLProbeConfig(
                initial_steps=probe_initial_steps,
                step_increment=int(
                    getattr(config, "scheduler_rl_probe_step_increment", rl_horizon)
                ),
                max_steps=probe_max_steps,
                required_passes=int(
                    getattr(config, "scheduler_rl_probe_required_passes", 2)
                ),
                episode_interval=int(
                    getattr(config, "scheduler_rl_probe_episode_interval", 1)
                ),
                min_progress_delta=float(
                    getattr(config, "scheduler_rl_probe_min_progress_delta", 0.01)
                ),
                expert_progress_fraction=float(
                    getattr(
                        config,
                        "scheduler_rl_probe_expert_progress_fraction",
                        0.8,
                    )
                ),
                reference_motion_floor_m=probe_reference_motion_floor_m,
                stationary_path_tolerance_m=float(
                    os.getenv(
                        "SCHEDULER_RL_PROBE_STATIONARY_PATH_TOLERANCE_M",
                        str(
                            getattr(
                                config,
                                "scheduler_rl_probe_stationary_path_tolerance_m",
                                0.005,
                            )
                        ),
                    )
                ),
                max_path_deviation=float(
                    getattr(config, "scheduler_rl_probe_max_path_deviation", 0.08)
                ),
                stall_steps=int(getattr(config, "scheduler_rl_probe_stall_steps", 10)),
                progress_epsilon=float(
                    getattr(config, "scheduler_rl_probe_progress_epsilon", 1e-4)
                ),
                progress_epsilon_m=float(
                    os.getenv(
                        "SCHEDULER_RL_PROBE_PROGRESS_EPSILON_M",
                        str(
                            getattr(
                                config,
                                "scheduler_rl_probe_progress_epsilon_m",
                                0.0001,
                            )
                        ),
                    )
                ),
                step_quantum=rl_horizon,
                off_path_decrement=int(
                    getattr(
                        config,
                        "scheduler_rl_probe_off_path_decrement",
                        rl_horizon,
                    )
                ),
                safety_decrement=int(
                    getattr(
                        config,
                        "scheduler_rl_probe_safety_decrement",
                        2 * rl_horizon,
                    )
                ),
            )
            rl_probe_controller = AdaptiveRLProbeController(rl_probe_config)
            probe_initial_search_steps = int(
                getattr(config, "scheduler_rl_probe_initial_search_steps", 1)
            )
            probe_max_index_advance_value = int(
                getattr(config, "scheduler_rl_probe_max_index_advance", 1)
            )
            probe_max_index_advance = (
                None
                if probe_max_index_advance_value == 0
                else probe_max_index_advance_value
            )
            configured_probe_max_arc_advance_ratio = getattr(
                config,
                "scheduler_rl_probe_max_arc_advance_ratio",
                None,
            )
            probe_max_arc_advance_ratio_env = os.getenv(
                "SCHEDULER_RL_PROBE_MAX_ARC_ADVANCE_RATIO"
            )
            probe_max_arc_advance_ratio = (
                float(probe_max_arc_advance_ratio_env)
                if probe_max_arc_advance_ratio_env is not None
                else (
                    None
                    if configured_probe_max_arc_advance_ratio is None
                    else float(configured_probe_max_arc_advance_ratio)
                )
            )
            probe_arc_advance_slack_m = float(
                os.getenv(
                    "SCHEDULER_RL_PROBE_ARC_ADVANCE_SLACK_M",
                    str(
                        getattr(
                            config,
                            "scheduler_rl_probe_arc_advance_slack_m",
                            0.0,
                        )
                    ),
                )
            )
            probe_max_index_advance_label = (
                "disabled"
                if probe_max_index_advance is None
                else str(probe_max_index_advance)
            )
            probe_max_arc_advance_ratio_label = (
                "disabled"
                if probe_max_arc_advance_ratio is None
                else f"{probe_max_arc_advance_ratio:.2f}"
            )
            probe_rotation_weight = float(
                getattr(config, "scheduler_rl_probe_rotation_weight", 0.01)
            )
            rl_probe_progress_estimator = ExpertTrajectoryProgressEstimator(
                trajectory_connector.demo_tcp_poses,
                lookahead=int(getattr(config, "scheduler_rl_probe_lookahead", 20)),
                initial_search_steps=probe_initial_search_steps,
                max_index_advance=probe_max_index_advance,
                rotation_weight=probe_rotation_weight,
                initial_index=probe_start_index,
                max_arc_advance_ratio=probe_max_arc_advance_ratio,
                arc_advance_slack_m=probe_arc_advance_slack_m,
                motion_epsilon_m=rl_probe_config.progress_epsilon_m,
            )
            if FLAGS.checkpoint_path is not None:
                rl_probe_state_path = os.path.join(
                    os.path.abspath(FLAGS.checkpoint_path),
                    "scheduler_rl_probe_state.pkl",
                )
                if (
                    FLAGS.resume_training or FLAGS.allow_existing_checkpoint_path
                ) and os.path.exists(rl_probe_state_path):
                    try:
                        rl_probe_controller.restore_state(
                            load_pickle_file(rl_probe_state_path)
                        )
                        print_green(
                            "Loaded RL probe state: "
                            f"budget={rl_probe_controller.budget_steps} "
                            f"pass_streak={rl_probe_controller.pass_streak}"
                        )
                    except Exception as exc:
                        print(
                            "Ignoring incompatible RL probe state "
                            f"{rl_probe_state_path}: {exc}",
                            flush=True,
                        )
        print(
            "[option scheduler] enabled; "
            f"mode={getattr(config, 'aia_ablation', None) or ('learned' if FLAGS.learned_option_scheduler else 'manual')}; "
            "initial option=RL; "
            "press 0=on/off, 1=RL, 2=trajectory correction, 3=CodePolicy; "
            "exploration_weights(RL,TRAJECTORY_CORRECTION,CODE_POLICY)="
            f"{scheduler_dqn_config.exploration_weights or 'uniform'}; "
            f"frozen_feature_dim={frozen_encoder.feature_dim} "
            f"scheduler_state_dim={scheduler_state_builder.state_dim} "
            f"encoder_warmup_ms={encoder_warmup_ms:.1f}",
            flush=True,
        )
        if rl_probe_controller is not None:
            print(
                "[rl probe] enabled; "
                f"initial_budget={rl_probe_controller.budget_steps} "
                f"increment={rl_probe_controller.config.step_increment} "
                f"max_budget={rl_probe_controller.config.max_steps} "
                f"max_budget_source={probe_max_steps_source} "
                f"expert_original_pose_count={expert_original_pose_count} "
                f"probe_start_index={probe_start_index} "
                f"probe_reference_pose_count={probe_reference_pose_count} "
                "skip_leading_stationary="
                f"{probe_skip_leading_stationary} "
                f"initial_search_steps={probe_initial_search_steps} "
                "max_index_advance="
                f"{probe_max_index_advance_label} "
                "max_arc_advance_ratio="
                f"{probe_max_arc_advance_ratio_label} "
                "arc_advance_slack_m="
                f"{probe_arc_advance_slack_m:.4f} "
                f"rotation_weight={probe_rotation_weight:.4f} "
                f"required_passes={rl_probe_controller.config.required_passes} "
                f"episode_interval={rl_probe_controller.config.episode_interval} "
                "expert_progress_fraction="
                f"{rl_probe_controller.config.expert_progress_fraction:.2f} "
                "reference_motion_floor_m="
                f"{rl_probe_controller.config.reference_motion_floor_m:.4f} "
                "stationary_path_tolerance_m="
                f"{rl_probe_controller.config.stationary_path_tolerance_m:.4f} "
                "progress_epsilon_m="
                f"{rl_probe_controller.config.progress_epsilon_m:.4f}",
                flush=True,
            )
        if policy_change_probe_controller is not None:
            print(
                "[policy-change probe] enabled; "
                f"regions={policy_change_probe_anchors.region_names} "
                f"region_starts={policy_change_probe_anchors.region_start_indices} "
                "anchors_per_region="
                f"{tuple(len(items) for items in policy_change_probe_anchors.anchor_observations)} "
                f"drift_threshold={policy_change_probe_controller.config.drift_threshold:.6f} "
                f"use_policy_drift={policy_change_probe_controller.config.use_policy_drift} "
                f"max_age_steps={policy_change_probe_controller.config.max_age_steps} "
                "budget="
                f"{policy_change_probe_controller.config.budget_steps}/"
                f"{policy_change_probe_controller.config.budget_window_steps} "
                f"rl_horizon={policy_change_probe_controller.config.rl_option_horizon} "
                "region_demo_steps="
                f"{policy_change_probe_anchors.region_transition_counts()} "
                "region_max_horizons="
                f"{policy_change_probe_controller.config.region_max_horizons} "
                f"initial_horizon={policy_change_probe_controller.config.initial_horizon_steps} "
                f"horizon_increment={policy_change_probe_controller.config.horizon_increment_steps} "
                f"required_passes={policy_change_probe_controller.config.required_passes} "
                f"state_feature_dim={policy_change_probe_controller.state_feature_dim} "
                "anchor_fingerprint="
                f"{policy_change_probe_anchors.fingerprint[:12]}",
                flush=True,
            )

    def project_rl_probe_progress():
        if rl_probe_progress_estimator is None:
            return None
        return rl_probe_progress_estimator.project(_get_env_curr_pose(env))

    def save_rl_probe_state():
        if rl_probe_state_path is not None:
            try:
                atomic_pickle_dump(
                    rl_probe_controller.state_dict(), rl_probe_state_path
                )
            except Exception as exc:
                print(f"[rl probe] state save failed: {exc}", flush=True)

    def handle_rl_probe_result(result, step):
        if result is None:
            return
        save_rl_probe_state()
        metrics = result.metrics()
        metrics.update(rl_probe_controller.schedule_metrics())
        metrics[f"rl_probe/reason/{result.reason}"] = 1
        try:
            client.request("send-stats", metrics)
        except Exception as exc:
            print(f"[rl probe] stats log failed: {exc}", flush=True)
        print(
            "[rl probe] finished "
            f"reason={result.reason} executed={result.executed_steps}/"
            f"{result.budget_before} progress={result.start_progress:.4f}->"
            f"{result.end_progress:.4f} delta={result.progress_delta:.4f} "
            f"required_delta={result.required_progress_delta:.4f} "
            f"ratio={result.progress_ratio:.3f} "
            f"target={result.expert_target_progress:.4f} "
            f"arc_m={result.start_arc_length_m:.4f}->"
            f"{result.end_arc_length_m:.4f} "
            f"motion_m={result.actual_motion_m:.4f}/"
            f"{result.required_motion_m:.4f} "
            f"expert_motion_m={result.expert_motion_m:.4f} "
            f"grading={result.grading_mode} "
            f"max_deviation={result.max_path_deviation:.4f} "
            f"passed={result.passed} pass_streak={result.pass_streak} "
            f"next_budget={result.budget_after}",
            flush=True,
        )

    def try_start_pending_rl_probe(step):
        if rl_probe_controller is None:
            return False
        if rl_probe_controller.active or rl_probe_controller.attempted_this_episode:
            return rl_probe_controller.active
        rl_probe_progress_estimator.reset()
        try:
            progress = project_rl_probe_progress()
        except Exception as exc:
            print(f"[rl probe] skipped: progress projection failed: {exc}", flush=True)
            return False
        milestone = rl_probe_progress_estimator.milestone_after_steps(
            progress.index,
            rl_probe_controller.budget_steps,
        )
        started = rl_probe_controller.start_episode(
            progress,
            expert_target_index=milestone.index,
            expert_target_progress=milestone.progress,
            expert_target_arc_length_m=milestone.arc_length_m,
        )
        if started:
            print(
                "[rl probe] episode started "
                f"budget={rl_probe_controller.budget_steps} "
                f"progress={progress.progress:.4f} "
                f"target_progress={milestone.progress:.4f} "
                "required_delta="
                f"{rl_probe_controller.required_progress_delta:.4f} "
                f"arc_m={progress.arc_length_m:.4f}->"
                f"{milestone.arc_length_m:.4f} "
                f"expert_motion_m={rl_probe_controller.expert_motion_m:.4f} "
                f"required_motion_m={rl_probe_controller.required_motion_m:.4f} "
                f"grading={rl_probe_controller.grading_mode} "
                "stationary_path_limit_m="
                f"{rl_probe_controller.stationary_path_limit_m:.4f} "
                f"deviation={progress.translation_distance:.4f}",
                flush=True,
            )
        else:
            print(
                "[rl probe] skipped: initial pose is outside expert corridor "
                f"deviation={progress.translation_distance:.4f} "
                f"limit={rl_probe_controller.config.max_path_deviation:.4f}",
                flush=True,
            )
        return started

    def start_rl_probe_episode(step):
        if rl_probe_controller is None:
            return False
        if rl_probe_controller.active:
            handle_rl_probe_result(rl_probe_controller.complete("episode_reset"), step)
        scheduled = rl_probe_controller.reset_episode()
        save_rl_probe_state()
        schedule_metrics = rl_probe_controller.schedule_metrics()
        try:
            client.request("send-stats", schedule_metrics)
        except Exception as exc:
            print(f"[rl probe] schedule log failed: {exc}", flush=True)
        print(
            "[rl probe] episode schedule "
            f"episode={rl_probe_controller.episode_counter} "
            f"scheduled={scheduled} "
            "non_probe_episodes_until_next="
            f"{rl_probe_controller.episodes_until_next_probe}",
            flush=True,
        )
        if not scheduled:
            return False
        return try_start_pending_rl_probe(step)

    def cancel_pending_rl_probe_episode():
        if (
            rl_probe_controller is not None
            and not rl_probe_controller.active
            and not rl_probe_controller.attempted_this_episode
        ):
            rl_probe_controller.cancel_episode()

    def complete_rl_probe(
        reason,
        step,
        *,
        success=False,
        off_path_decrement=False,
        safety_decrement=False,
    ):
        if rl_probe_controller is None or not rl_probe_controller.active:
            return None
        progress = None
        try:
            progress = project_rl_probe_progress()
        except Exception as exc:
            print(f"[rl probe] final progress projection failed: {exc}", flush=True)
        result = rl_probe_controller.complete(
            reason,
            progress=progress,
            success=success,
            off_path_decrement=off_path_decrement,
            safety_decrement=safety_decrement,
        )
        handle_rl_probe_result(result, step)
        return result

    def scheduler_action_mask(option_obs):
        if not scheduler_configured:
            return np.ones((len(OptionID),), dtype=bool)
        return code_policy_episode_guard.apply_action_mask(
            np.asarray(
                [
                    bool(
                        getattr(
                            config, "scheduler_allowed_options", (True, True, True)
                        )[int(option_id)]
                    )
                    and manual_options[option_id].available(option_obs)
                    for option_id in OptionID
                ],
                dtype=bool,
            )
        )

    def update_policy_change_probe_metrics(decision):
        if decision is None:
            return
        policy_change_probe_episode_metrics.update(decision.metrics())
        policy_change_probe_episode_metrics["policy_change_probe/base_option"] = float(
            -1 if active_base_option_id is None else int(active_base_option_id)
        )

    def stop_active_option(
        reason=None,
        *,
        end_obs=None,
        terminal_reward=None,
        terminated_override=None,
        truncated_override=None,
    ):
        nonlocal active_option
        nonlocal active_scheduler_type
        nonlocal scheduler_pending_transition
        nonlocal option_motion_accumulator
        nonlocal active_policy_change_probe_decision
        nonlocal active_autonomous_region
        nonlocal active_autonomous_signature
        nonlocal active_base_option_id
        if active_option is None or not active_option.active:
            active_option = None
            return None
        clear_scheduler_boundary_cache()
        next_state_ms = None
        next_mask_ms = None
        completed_target = (
            active_option.target
            if active_option.option_id is OptionID.TRAJECTORY_CORRECTION
            else None
        )
        result = active_option.stop(reason)
        if (
            trajectory_connector is not None
            and completed_target is not None
            and result.termination_reason is TerminationReason.TARGET_REACHED
        ):
            trajectory_connector.commit_target(completed_target)

        if result.duration > 0:
            if (
                scheduler_pending_transition is None
                or option_motion_accumulator is None
            ):
                raise RuntimeError(
                    "Active Option is missing Scheduler transition context"
                )
            task_return_override = None
            if terminal_reward is not None:
                task_return_override = float(result.discounted_return) + (
                    float(config.discount) ** int(result.duration)
                ) * float(terminal_reward)
            scheduler_reward = scheduler_reward_config.reward(
                result,
                task_return_override=task_return_override,
            )
            motion_summary = option_motion_accumulator.finish()
            scheduler_history.append(result, scheduler_reward, motion_summary)
            terminal = bool(
                result.episode_terminated
                if terminated_override is None
                else terminated_override
            ) or bool(
                result.episode_truncated
                if truncated_override is None
                else truncated_override
            )
            if (
                result.option_id is OptionID.RL
                and policy_change_probe_controller is not None
            ):
                if active_autonomous_region is None or (
                    policy_change_probe_controller.config.use_policy_drift
                    and active_autonomous_signature is None
                ):
                    raise RuntimeError(
                        "RL Option is missing policy-change Probe start evidence"
                    )
                autonomous_return = (
                    float(result.discounted_return)
                    if task_return_override is None
                    else float(task_return_override)
                )
                success = bool(
                    terminal
                    and (
                        float(terminal_reward) > 0.5
                        if terminal_reward is not None
                        else autonomous_return > 0.0
                    )
                )
                final_progress = current_policy_change_progress(
                    obs if end_obs is None else end_obs
                )
                final_region = policy_change_probe_anchors.region_for_progress_index(
                    final_progress.index
                )
                outcome = policy_change_probe_controller.record_autonomous_execution(
                    region=active_autonomous_region,
                    signature=active_autonomous_signature,
                    start_step=result.start_step,
                    duration=result.duration,
                    task_return=autonomous_return,
                    termination_reason=result.termination_reason.value,
                    success=success,
                    episode_terminated=bool(
                        result.episode_terminated
                        if terminated_override is None
                        else terminated_override
                    ),
                    episode_truncated=bool(
                        result.episode_truncated
                        if truncated_override is None
                        else truncated_override
                    ),
                    probe_decision=(
                        active_policy_change_probe_decision
                        if active_scheduler_type == "policy_change_probe"
                        else None
                    ),
                    final_region=final_region,
                    path_deviation_m=float(final_progress.translation_distance),
                )
                save_policy_change_probe_state()
                policy_change_probe_episode_metrics.update(
                    {
                        "policy_change_probe/last_autonomous_duration": float(
                            outcome.duration
                        ),
                        "policy_change_probe/last_autonomous_return": float(
                            outcome.task_return
                        ),
                        "policy_change_probe/last_autonomous_success": float(
                            outcome.success
                        ),
                        "policy_change_probe/last_autonomous_was_probe": float(
                            outcome.was_probe
                        ),
                        "policy_change_probe/session_active": float(
                            policy_change_probe_controller.probe_session_active
                        ),
                        "policy_change_probe/current_horizon_steps": float(
                            policy_change_probe_controller.current_horizon_steps(
                                active_autonomous_region
                            )
                        ),
                        "policy_change_probe/pass_streak": float(
                            policy_change_probe_controller.pass_streak(
                                active_autonomous_region
                            )
                        ),
                    }
                )
                print(
                    "[policy-change probe] autonomous chunk "
                    f"region={policy_change_probe_controller.config.region_names[active_autonomous_region]} "
                    f"duration={outcome.duration} return={outcome.task_return:.4f} "
                    f"reason={outcome.termination_reason} probe={outcome.was_probe} "
                    f"session_active={policy_change_probe_controller.probe_session_active} "
                    f"horizon={policy_change_probe_controller.current_horizon_steps(active_autonomous_region)}",
                    flush=True,
                )
            final_obs = obs if end_obs is None else end_obs
            next_state_started = time.perf_counter()
            next_scheduler_state = scheduler_state_builder.build(
                final_obs, scheduler_history
            )
            next_state_ms = 1000.0 * (time.perf_counter() - next_state_started)
            next_mask_started = time.perf_counter()
            next_action_mask = (
                np.zeros((len(OptionID),), dtype=bool)
                if terminal
                else scheduler_action_mask(final_obs)
            )
            next_mask_ms = 1000.0 * (time.perf_counter() - next_mask_started)
            scheduler_transition = scheduler_transition_builder.finish(
                scheduler_pending_transition,
                result,
                next_scheduler_state,
                next_action_mask,
                scheduler_reward=scheduler_reward,
                terminated_override=terminated_override,
                truncated_override=truncated_override,
            )
            scheduler_transition = dict(scheduler_transition)
            scheduler_transition["option_name"] = result.option_id.name
            scheduler_transition[
                "termination_reason_name"
            ] = result.termination_reason.value
            scheduler_transition["scheduler_type"] = active_scheduler_type
            if active_base_option_id is not None:
                scheduler_transition["base_option_id"] = int(active_base_option_id)
                scheduler_transition["base_option_name"] = active_base_option_id.name
            if active_policy_change_probe_decision is not None:
                decision = active_policy_change_probe_decision
                scheduler_transition.update(
                    {
                        "policy_change_probe_requested": bool(decision.requested),
                        "policy_change_probe_override": bool(decision.override),
                        "policy_change_probe_reason": decision.reason,
                        "policy_change_probe_trigger_reasons": decision.trigger_reasons,
                        "policy_change_probe_region": int(decision.region),
                        "policy_change_probe_region_name": decision.region_name,
                        "policy_change_probe_age_steps": decision.age_steps,
                        "policy_change_probe_drift": (
                            None
                            if decision.drift is None
                            else float(decision.drift.combined_skl)
                        ),
                        "policy_change_probe_continuous_drift": (
                            None
                            if decision.drift is None
                            else float(decision.drift.continuous_skl)
                        ),
                        "policy_change_probe_discrete_drift": (
                            None
                            if decision.drift is None
                            else float(decision.drift.discrete_skl)
                        ),
                        "policy_change_probe_remaining_budget_steps": int(
                            decision.remaining_budget_steps
                        ),
                        "policy_change_probe_budget_window_index": int(
                            decision.budget_window_index
                        ),
                    }
                )
            scheduler_data_store.insert(copy.deepcopy(scheduler_transition))
            current_scheduler_trajectory.append(scheduler_transition)
            if not terminal:
                # In an SMDP, the completed Option's z_end is exactly the next
                # Option's z_start. Reuse it instead of running the frozen
                # visual encoder and availability providers a second time.
                scheduler_boundary_cache["observation"] = final_obs
                scheduler_boundary_cache["state"] = next_scheduler_state
                scheduler_boundary_cache["action_mask"] = next_action_mask
        elif scheduler_pending_transition is not None:
            if (
                active_scheduler_type == "policy_change_probe"
                and policy_change_probe_controller is not None
                and active_policy_change_probe_decision is not None
            ):
                policy_change_probe_controller.cancel_probe_reservation(
                    active_policy_change_probe_decision,
                    start_step=result.start_step,
                )
                save_policy_change_probe_state()
            print(
                "[option scheduler] skipped zero-duration high-level transition",
                flush=True,
            )

        scheduler_pending_transition = None
        option_motion_accumulator = None
        active_policy_change_probe_decision = None
        active_autonomous_region = None
        active_autonomous_signature = None
        active_base_option_id = None
        print(
            "[option scheduler] stop "
            f"option={result.option_id.name} duration={result.duration} "
            f"reason={result.termination_reason.value}"
            + ("" if next_state_ms is None else f" next_state_ms={next_state_ms:.1f}")
            + ("" if next_mask_ms is None else f" next_mask_ms={next_mask_ms:.1f}"),
            flush=True,
        )
        active_option = None
        active_scheduler_type = "disabled"
        return result

    def clear_scheduler_session_state():
        nonlocal selected_option_id
        nonlocal manual_option_override_pending
        nonlocal scheduler_pending_transition
        nonlocal option_motion_accumulator
        if not scheduler_configured:
            return
        selected_option_id = OptionID.RL
        manual_option_override_pending = False
        scheduler_history.reset()
        scheduler_pending_transition = None
        option_motion_accumulator = None
        clear_scheduler_boundary_cache()
        if (
            policy_change_probe_controller is not None
            and policy_change_probe_controller.probe_session_active
        ):
            policy_change_probe_controller.abort_probe_session(
                end_step=last_seen_step,
                termination_reason="scheduler_session_reset",
            )
            save_policy_change_probe_state()
        trajectory_connector.reset()
        manual_options[OptionID.CODE_POLICY].reset_episode()

    def reset_manual_option_state(current_obs=None, step=None):
        if not scheduler_configured:
            return
        if rl_probe_controller is not None and rl_probe_controller.active:
            complete_rl_probe(
                "episode_reset",
                last_seen_step if step is None else step,
            )
        stop_active_option(TerminationReason.SAFETY_INTERRUPTION)
        clear_scheduler_session_state()
        if current_obs is not None and rl_probe_controller is not None:
            if rl_probe_controller.mark_episode_completed():
                save_rl_probe_state()
        if current_obs is not None and scheduler_enabled:
            start_rl_probe_episode(last_seen_step if step is None else step)

    def set_scheduler_runtime_enabled(enabled, *, current_obs):
        nonlocal scheduler_enabled
        if not scheduler_configured:
            return
        enabled = bool(enabled)
        if enabled == scheduler_enabled:
            return

        if not enabled:
            # The raw-RL interval is outside the Scheduler SMDP. End the
            # current high-level sequence without bootstrapping across it.
            complete_rl_probe("scheduler_disabled", last_seen_step)
            cancel_pending_rl_probe_episode()
            stop_active_option(
                TerminationReason.SCHEDULER_SWITCH,
                end_obs=current_obs,
                terminated_override=False,
                truncated_override=True,
            )
            scheduler_enabled = False
            clear_scheduler_session_state()
            action_filter.reset()
            print(
                "[option scheduler] runtime disabled by key 0; " "control=raw_rl",
                flush=True,
            )
            return

        # Observations collected while disabled do not have matching Option
        # history, so start a fresh high-level sequence when re-enabled.
        clear_scheduler_session_state()
        scheduler_enabled = True
        action_filter.reset()
        print(
            "[option scheduler] runtime enabled by key 0; "
            f"mode={getattr(config, 'aia_ablation', None) or ('learned' if FLAGS.learned_option_scheduler else 'manual')}",
            flush=True,
        )

    if scheduler_enabled:
        start_rl_probe_episode(start_step)

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    try:
        for step in pbar:
            last_seen_step = step
            timer.tick("total")
            # Policy-change evidence is defined for one immutable policy over a
            # complete Option. Other runs retain the legacy periodic apply.
            if not (
                scheduler_enabled
                and policy_change_probe_controller is not None
                and (
                    active_option is not None
                    or policy_change_probe_controller.probe_session_active
                )
            ):
                apply_pending_network()

            if scheduler_configured:
                if manual_labeler.consume_scheduler_toggle():
                    set_scheduler_runtime_enabled(
                        not scheduler_enabled,
                        current_obs=obs,
                    )

                option_request = manual_labeler.consume_option_request()
                if option_request is not None:
                    if not scheduler_enabled:
                        print(
                            "[option scheduler] ignored option key while runtime disabled; "
                            "press 0 to enable",
                            flush=True,
                        )
                        option_request = None

                if option_request is not None:
                    requested_option = manual_options[option_request]
                    available_actions = scheduler_action_mask(obs)
                    if not available_actions[int(option_request)]:
                        print(
                            "[option scheduler] rejected unavailable or masked "
                            f"option={option_request.name}",
                            flush=True,
                        )
                    else:
                        probe_was_active = bool(
                            rl_probe_controller is not None
                            and rl_probe_controller.active
                        )
                        if active_option is not None and (
                            active_option.option_id is not option_request
                            or probe_was_active
                        ):
                            stop_active_option(TerminationReason.SCHEDULER_SWITCH)
                            action_filter.reset()
                        elif (
                            policy_change_probe_controller is not None
                            and policy_change_probe_controller.probe_session_active
                        ):
                            policy_change_probe_controller.abort_probe_session(
                                end_step=step,
                                termination_reason="manual_override",
                            )
                            save_policy_change_probe_state()
                        complete_rl_probe("manual_override", step)
                        cancel_pending_rl_probe_episode()
                        selected_option_id = option_request
                        manual_option_override_pending = True
                        print(
                            "[option scheduler] manual override selected "
                            f"option={selected_option_id.name}",
                            flush=True,
                        )

            manual_label = (
                "primitive_intervention"
                if primitive_intervention_pending
                else manual_labeler.consume()
            )
            primitive_intervention_pending = False
            if manual_label == "primitive_intervention":
                print("[actor primitive intervention] requested by key i", flush=True)
                complete_rl_probe(
                    "primitive_intervention",
                    step,
                    safety_decrement=True,
                )
                cancel_pending_rl_probe_episode()
                if scheduler_enabled:
                    stop_active_option(TerminationReason.HUMAN_INTERVENTION)
                    selected_option_id = OptionID.RL
                action_filter.reset()
                _call_env_chain(env, "clear_intervention")
                try:
                    primitive_plan = _call_env_chain_result(
                        env, "plan_intervention_primitive"
                    )
                    (
                        obs,
                        primitive_terminal,
                        primitive_manual_label,
                    ) = execute_primitive_plan_through_env_steps(
                        obs, primitive_plan, step
                    )
                    print(
                        "[actor primitive intervention] done "
                        f"execute={bool((primitive_plan or {}).get('execute', False))}",
                        flush=True,
                    )
                    if primitive_manual_label is not None:
                        obs = finish_manual_label_episode(
                            obs, primitive_manual_label, step
                        )
                        reset_manual_option_state(obs, step)
                    elif primitive_terminal:
                        obs = finish_terminal_episode(step, {})
                        reset_manual_option_state(obs, step)
                except Exception as exc:
                    print(f"[actor primitive intervention] failed: {exc}", flush=True)
                manual_labeler.clear()
                timer.tock("total")
                continue

            if manual_label is not None:
                if not current_trajectory:
                    print(
                        f"ignored stale manual {manual_label} label after reset",
                        flush=True,
                    )
                    timer.tock("total")
                    continue
                if scheduler_enabled:
                    terminal_reward = 1.0 if manual_label == "success" else 0.0
                    stop_active_option(
                        TerminationReason.EPISODE_TERMINATED,
                        end_obs=obs,
                        terminal_reward=terminal_reward,
                        terminated_override=True,
                        truncated_override=False,
                    )
                complete_rl_probe(
                    f"manual_{manual_label}",
                    step,
                    success=manual_label == "success",
                )
                obs = finish_manual_label_episode(obs, manual_label, step)
                reset_manual_option_state(obs, step)
                timer.tock("total")
                continue

            with timer.context("sample_actions"):
                if scheduler_enabled:
                    if active_option is None:
                        # Option boundaries are safe synchronization points for
                        # the latest high- and low-level network parameters.
                        probe_session_active = bool(
                            policy_change_probe_controller is not None
                            and policy_change_probe_controller.probe_session_active
                        )
                        params_changed = (
                            False
                            if probe_session_active
                            else apply_pending_network(force=True)
                        )
                        signatures_changed = (
                            False
                            if probe_session_active
                            else refresh_policy_change_probe_signatures()
                        )
                        if params_changed or signatures_changed:
                            clear_scheduler_boundary_cache()
                        cache_hit = bool(
                            scheduler_boundary_cache["state"] is not None
                            and scheduler_boundary_cache["action_mask"] is not None
                            and scheduler_boundary_cache["observation"] is obs
                        )
                        if cache_hit:
                            scheduler_state = scheduler_boundary_cache["state"]
                            available_actions = scheduler_boundary_cache["action_mask"]
                            scheduler_state_source = "boundary_cache"
                            scheduler_state_ms = 0.0
                            scheduler_mask_ms = 0.0
                        else:
                            scheduler_state_started = time.perf_counter()
                            scheduler_state = scheduler_state_builder.build(
                                obs, scheduler_history
                            )
                            scheduler_state_ms = 1000.0 * (
                                time.perf_counter() - scheduler_state_started
                            )
                            scheduler_mask_started = time.perf_counter()
                            available_actions = scheduler_action_mask(obs)
                            scheduler_mask_ms = 1000.0 * (
                                time.perf_counter() - scheduler_mask_started
                            )
                            scheduler_state_source = "encoded"
                        clear_scheduler_boundary_cache()
                        behavior_prob = 1.0
                        q_values = None
                        scheduler_policy_ms = 0.0
                        active_policy_change_probe_decision = None
                        active_autonomous_region = None
                        active_autonomous_signature = None
                        active_base_option_id = None
                        try_start_pending_rl_probe(step)
                        rl_probe_forces_rl = False
                        if (
                            rl_probe_controller is not None
                            and rl_probe_controller.active
                        ):
                            rl_probe_forces_rl = bool(
                                available_actions[int(OptionID.RL)]
                            )
                            if not rl_probe_forces_rl:
                                complete_rl_probe(
                                    "rl_unavailable",
                                    step,
                                    safety_decrement=True,
                                )
                        if manual_option_override_pending:
                            active_scheduler_type = "manual_override"
                            manual_option_override_pending = False
                        elif rl_probe_forces_rl:
                            selected_option_id = OptionID.RL
                            behavior_prob = 1.0
                            active_scheduler_type = "rl_probe"
                        elif fixed_option_scheduler:
                            distribution = probabilities(
                                scheduler_dqn_config.exploration_weights,
                                available_actions,
                            )
                            # Step-keyed RNG avoids restarting the random sequence on resume.
                            fixed_rng = np.random.default_rng(
                                [FLAGS.seed, step, 0xAB1A]
                            )
                            selected_action = int(
                                fixed_rng.choice(len(OptionID), p=distribution)
                            )
                            selected_option_id = OptionID(selected_action)
                            behavior_prob = distribution[selected_action]
                            active_scheduler_type = "fixed_rule"
                        elif FLAGS.learned_option_scheduler and scheduler_policy_ready:
                            epsilon = scheduler_dqn_config.epsilon(
                                scheduler_update_step
                            )
                            sampling_rng, scheduler_key = jax.random.split(sampling_rng)
                            scheduler_policy_started = time.perf_counter()
                            (
                                selected_action,
                                behavior_prob,
                                q_values,
                            ) = scheduler_agent.sample_action(
                                scheduler_state,
                                available_actions,
                                seed=scheduler_key,
                                epsilon=epsilon,
                            )
                            scheduler_policy_ms = 1000.0 * (
                                time.perf_counter() - scheduler_policy_started
                            )
                            selected_option_id = OptionID(selected_action)
                            active_scheduler_type = "learned"
                        elif FLAGS.learned_option_scheduler:
                            selected_option_id = OptionID.RL
                            active_scheduler_type = "policy_unavailable_fallback"
                        else:
                            active_scheduler_type = "manual"

                        active_base_option_id = selected_option_id
                        if (
                            policy_change_probe_controller is not None
                            and active_scheduler_type in {"learned", "fixed_rule"}
                        ):
                            probe_progress = current_policy_change_progress(obs)
                            probe_region = (
                                policy_change_probe_anchors.region_for_progress_index(
                                    probe_progress.index
                                )
                            )
                            active_policy_change_probe_decision = policy_change_probe_controller.decide(
                                base_is_rl=(selected_option_id is OptionID.RL),
                                rl_available=bool(available_actions[int(OptionID.RL)]),
                                region=probe_region,
                                step=step,
                                remaining_region_steps=(
                                    policy_change_probe_anchors.remaining_region_steps(
                                        probe_progress.index
                                    )
                                ),
                            )
                            update_policy_change_probe_metrics(
                                active_policy_change_probe_decision
                            )
                            decision = active_policy_change_probe_decision
                            print(
                                "[policy-change probe] decision "
                                f"base={selected_option_id.name} "
                                f"region={decision.region_name} "
                                f"requested={decision.requested} "
                                f"override={decision.override} "
                                f"reason={decision.reason} "
                                f"triggers={decision.trigger_reasons} "
                                f"age={decision.age_steps} "
                                "drift="
                                f"{None if decision.drift is None else round(decision.drift.combined_skl, 6)} "
                                f"budget_remaining={decision.remaining_budget_steps}",
                                f"session_horizon={decision.session_horizon_steps} "
                                f"region_max={decision.region_max_horizon_steps} "
                                f"continuation={decision.session_continuation}",
                                flush=True,
                            )
                            if decision.override:
                                selected_option_id = OptionID.RL
                                behavior_prob = 1.0
                                active_scheduler_type = "policy_change_probe"
                            elif decision.session_continuation:
                                policy_change_probe_controller.abort_probe_session(
                                    end_step=step,
                                    termination_reason=decision.reason,
                                )
                                save_policy_change_probe_state()

                        option = manual_options[selected_option_id]
                        if not scheduler_action_mask(obs)[int(selected_option_id)]:
                            print(
                                "[option scheduler] selected option became "
                                f"unavailable: {selected_option_id.name}; falling back to RL",
                                flush=True,
                            )
                            selected_option_id = OptionID.RL
                            option = manual_options[selected_option_id]
                            behavior_prob = 1.0
                            active_scheduler_type = "availability_fallback"
                        if (
                            policy_change_probe_controller is not None
                            and selected_option_id is OptionID.RL
                        ):
                            active_autonomous_region = current_policy_change_region(obs)
                            active_autonomous_signature = (
                                policy_change_probe_controller.current_signature(
                                    active_autonomous_region
                                )
                            )
                        if active_scheduler_type == "policy_change_probe":
                            policy_change_probe_controller.reserve_probe(
                                active_policy_change_probe_decision,
                                start_step=step,
                            )
                            save_policy_change_probe_state()
                        pending_transition = scheduler_transition_builder.begin(
                            scheduler_state,
                            option.option_id,
                            available_actions,
                            behavior_prob=behavior_prob,
                            start_step=step,
                            rl_policy_version=rl_policy_version,
                        )
                        motion_accumulator = OptionMotionAccumulator(obs)
                        option_start_started = time.perf_counter()
                        option.start(obs, global_step=step)
                        option_start_ms = 1000.0 * (
                            time.perf_counter() - option_start_started
                        )
                        active_option = option
                        code_policy_episode_guard.mark_option_started(
                            active_option.option_id
                        )
                        scheduler_pending_transition = pending_transition
                        option_motion_accumulator = motion_accumulator
                        print(
                            "[option scheduler] start "
                            f"option={active_option.option_id.name} step={step} "
                            f"source={active_scheduler_type} "
                            f"behavior_prob={behavior_prob:.4f}"
                            f" state_source={scheduler_state_source}"
                            f" state_ms={scheduler_state_ms:.1f}"
                            f" mask_ms={scheduler_mask_ms:.1f}"
                            f" policy_ms={scheduler_policy_ms:.1f}"
                            f" option_start_ms={option_start_ms:.1f}"
                            + (
                                ""
                                if q_values is None
                                else " q=" + np.array2string(q_values, precision=4)
                            ),
                            flush=True,
                        )
                    option_step_context["step"] = step
                    executed_option = active_option
                    option_metadata = {
                        "option_id": int(executed_option.option_id),
                        "option_name": executed_option.option_id.name,
                        "option_start_step": step - executed_option.duration,
                        "option_step": executed_option.duration,
                        "scheduler_type": active_scheduler_type,
                    }
                    if active_base_option_id is not None:
                        option_metadata.update(
                            {
                                "base_option_id": int(active_base_option_id),
                                "base_option_name": active_base_option_id.name,
                            }
                        )
                    if active_policy_change_probe_decision is not None:
                        decision = active_policy_change_probe_decision
                        option_metadata.update(
                            {
                                "policy_change_probe_requested": bool(
                                    decision.requested
                                ),
                                "policy_change_probe_override": bool(decision.override),
                                "policy_change_probe_reason": decision.reason,
                                "policy_change_probe_region": int(decision.region),
                                "policy_change_probe_region_name": (
                                    decision.region_name
                                ),
                                "policy_change_probe_age_steps": (decision.age_steps),
                                "policy_change_probe_drift": (
                                    None
                                    if decision.drift is None
                                    else float(decision.drift.combined_skl)
                                ),
                                "policy_change_probe_remaining_budget_steps": int(
                                    decision.remaining_budget_steps
                                ),
                            }
                        )
                    if (
                        active_scheduler_type == "rl_probe"
                        and rl_probe_controller is not None
                    ):
                        option_metadata.update(
                            {
                                "rl_probe_budget_steps": (
                                    rl_probe_controller.budget_steps
                                ),
                                "rl_probe_remaining_steps": (
                                    rl_probe_controller.remaining_steps
                                ),
                            }
                        )
                    actions = executed_option.act(obs)
                else:
                    executed_option = None
                    option_metadata = None
                    if step < config.random_steps:
                        actions = env.action_space.sample()
                    else:
                        sampling_rng, key = jax.random.split(sampling_rng)
                        actions = agent.sample_actions(
                            observations=jax.device_put(obs),
                            seed=key,
                            argmax=False,
                        )
                        actions = np.asarray(jax.device_get(actions))
                        actions = action_filter(actions)

            # Step environment
            with timer.context("step_env"):
                next_obs, reward, done, truncated, info = env.step(actions)
                code_policy_episode_guard.record_env_step()
                if "left" in info:
                    info.pop("left")
                if "right" in info:
                    info.pop("right")

                if "intervene_action" in info:
                    actions = info.pop("intervene_action")
                    actions = clip_action_to_space(actions, env.action_space)
                    print("\n\n INTERVENING: ", actions)
                    intervention_steps += 1
                    if not already_intervened:
                        intervention_count += 1
                    already_intervened = True
                    current_intervention = True
                    trajectory_had_intervention = True
                else:
                    actions = info.pop("executed_action", actions)
                    already_intervened = False
                    current_intervention = False

                option_demo_eligible = bool(
                    executed_option is not None
                    and executed_option.option_id in demo_buffer_option_ids
                )
                demo_eligible = bool(current_intervention or option_demo_eligible)
                if current_intervention:
                    demo_source = "human_intervention"
                elif option_demo_eligible:
                    demo_source = f"option:{executed_option.option_id.name}"
                else:
                    demo_source = None

                if (
                    executed_option is not None
                    and executed_option.option_id is OptionID.CODE_POLICY
                ):
                    print(
                        "[CodePolicy executed] "
                        f"action={np.array2string(np.asarray(actions), precision=5)} "
                        f"intervention_override={current_intervention} "
                        f"demo_eligible={demo_eligible}",
                        flush=True,
                    )

                manual_label = manual_labeler.consume()
                if manual_label == "success":
                    prepare_manual_reset(env, action_filter)
                    reward = 1.0
                    done = True
                    truncated = False
                    info["succeed"] = True
                    info.setdefault("episode", {})
                    print("manual success marked; resetting environment")
                elif manual_label == "failure":
                    prepare_manual_reset(env, action_filter)
                    reward = 0.0
                    done = True
                    truncated = False
                    info["succeed"] = False
                    info.setdefault("episode", {})
                    print("manual failure marked; resetting environment")
                elif manual_label == "primitive_intervention":
                    primitive_intervention_pending = True
                    print(
                        "[actor primitive intervention] queued after current env.step",
                        flush=True,
                    )

                terminal = bool(done or truncated)
                if terminal:
                    info.setdefault("succeed", bool(reward > 0.0 and not truncated))
                    info.setdefault("episode", {})
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                terminal_demo_bridge = should_add_terminal_demo_bridge(
                    current_trajectory,
                    reward,
                    terminal,
                    already_demo_eligible=demo_eligible,
                )
                if terminal_demo_bridge:
                    demo_eligible = True
                    demo_source = "terminal_bridge_after_demo"
                    print(
                        "[demo terminal bridge] added post-step success "
                        "transition to demo buffer",
                        flush=True,
                    )
                running_return += reward

                schedule_intervention = bool(
                    not current_intervention
                    and executed_option is not None
                    and executed_option.option_id is not OptionID.RL
                )
                info["human_intervention"] = bool(current_intervention)
                info["other_strategy_intervention"] = schedule_intervention
                if current_intervention:
                    info["intervention_source"] = "human"
                elif schedule_intervention:
                    info[
                        "intervention_source"
                    ] = f"schedule:{executed_option.option_id.name.lower()}"
                if option_metadata is not None:
                    info.update(option_metadata)

                if trajectory_connector is not None:
                    actual_trajectory_progress = trajectory_connector.observe(
                        _get_env_curr_pose(env)
                    )
                    info["trajectory_progress_source"] = "actual_tcp_state"
                    info["trajectory_progress_index"] = int(
                        actual_trajectory_progress.index
                    )
                    info["trajectory_progress"] = float(
                        actual_trajectory_progress.progress
                    )
                    info["trajectory_progress_arc_m"] = float(
                        actual_trajectory_progress.arc_length_m
                    )
                    if (
                        policy_change_probe_controller is not None
                        and policy_change_probe_controller.probe_session_active
                    ):
                        policy_change_probe_controller.observe_probe_path(
                            actual_trajectory_progress.translation_distance
                        )

                option_result = None
                if executed_option is not None:
                    executed_option.observe(
                        next_obs,
                        reward,
                        done,
                        truncated,
                        info,
                    )
                    option_motion_accumulator.observe(next_obs)
                    option_metadata["option_duration"] = executed_option.duration

                    if current_intervention or primitive_intervention_pending:
                        option_result = stop_active_option(
                            TerminationReason.HUMAN_INTERVENTION,
                            end_obs=next_obs,
                        )
                        selected_option_id = OptionID.RL
                    elif (
                        active_scheduler_type == "policy_change_probe"
                        and active_autonomous_region is not None
                        and current_policy_change_region(next_obs)
                        != active_autonomous_region
                    ):
                        option_result = stop_active_option(
                            TerminationReason.SCHEDULER_SWITCH,
                            end_obs=next_obs,
                        )
                    elif executed_option.should_terminate():
                        option_result = stop_active_option(
                            end_obs=next_obs,
                        )
                        if executed_option.option_id is not OptionID.RL:
                            selected_option_id = OptionID.RL

                    if option_result is not None:
                        option_metadata[
                            "option_termination_reason"
                        ] = option_result.termination_reason.value

                if (
                    option_metadata is not None
                    and option_metadata.get("scheduler_type") == "rl_probe"
                    and rl_probe_controller is not None
                    and rl_probe_controller.active
                ):
                    try:
                        rl_probe_progress = project_rl_probe_progress()
                    except Exception as exc:
                        print(
                            f"[rl probe] progress projection failed: {exc}",
                            flush=True,
                        )
                        probe_result = rl_probe_controller.complete(
                            "progress_projection_failed",
                            safety_decrement=True,
                        )
                    else:
                        probe_result = rl_probe_controller.observe_rl_step(
                            rl_probe_progress,
                            success=bool(
                                terminal
                                and info.get(
                                    "succeed",
                                    reward > 0.0 and not truncated,
                                )
                            ),
                            terminal=terminal,
                            intervention=bool(
                                current_intervention or primitive_intervention_pending
                            ),
                        )
                    handle_rl_probe_result(probe_result, step)

                transition = dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - float(done),
                    dones=terminal,
                    infos=copy.deepcopy(info),
                )
                if "grasp_penalty" in info:
                    transition["grasp_penalty"] = info["grasp_penalty"]

                transition = mark_transition_intervention(
                    transition, current_intervention, trajectory_had_intervention
                )
                transition = mark_transition_demo_eligibility(
                    transition, demo_eligible, demo_source
                )
                data_store.insert(transition)
                logged_transition = copy.deepcopy(transition)
                if option_metadata is not None:
                    logged_transition.setdefault("infos", {}).update(option_metadata)
                current_trajectory.append(logged_transition)
                if demo_eligible:
                    intvn_data_store.insert(copy.deepcopy(transition))

                obs = next_obs
                sync_actor_data(step)

                if done or truncated:
                    info.setdefault("episode", {})
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                    pbar.set_description(f"last return: {running_return}")
                    dump_trajectory(current_trajectory, step)
                    dump_scheduler_trajectory(step)
                    current_trajectory = []
                    sync_actor_data(step, force=True)
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
                    trajectory_had_intervention = False
                    prepare_manual_reset(env, action_filter)
                    finish_code_policy_guard_episode()
                    print("[actor] calling env.reset", flush=True)
                    obs, _ = env.reset()
                    manual_labeler.clear()
                    print("[actor] env.reset returned", flush=True)
                    action_filter.reset()
                    reset_manual_option_state(obs, step)

            if (
                config.buffer_period > 0
                and step - last_buffer_dump_step >= config.buffer_period
            ):
                dump_actor_buffers(step)

            timer.tock("total")

            # Keep the real-time actor loop free of blocking trainer requests.
            # Episode stats are sent after reset, where a short trainer timeout is less harmful.
    finally:
        if scheduler_configured:
            try:
                complete_rl_probe("actor_shutdown", last_seen_step)
                stop_active_option(TerminationReason.SAFETY_INTERRUPTION)
            except Exception as exc:
                print(f"option scheduler cleanup failed: {exc}")
        try:
            pbar.close()
        except Exception:
            pass
        try:
            dump_scheduler_trajectory(last_seen_step, final=True)
        except Exception as exc:
            print(f"final scheduler buffer dump failed: {exc}")
        try:
            dump_actor_buffers(last_seen_step, final=True)
        except Exception as exc:
            print(f"final buffer dump failed: {exc}")
        try:
            sync_actor_data(last_seen_step, force=True)
        except Exception as exc:
            print(f"final actor sync failed: {exc}")
        try:
            manual_labeler.stop()
        except Exception:
            pass
        try:
            env.close()
        except Exception as exc:
            print(f"actor cleanup failed: {exc}")


##############################################################################


def learner(
    rng,
    agent,
    scheduler_agent,
    scheduler_dqn_config,
    replay_buffer,
    demo_buffer,
    scheduler_replay_buffer,
    wandb_logger=None,
):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = get_resume_step(FLAGS.checkpoint_path) if FLAGS.resume_training else 0
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.register_data_store("scheduler_env", scheduler_replay_buffer)
    server.start(threaded=True)

    def scheduler_step_value():
        return int(np.asarray(jax.device_get(scheduler_agent.state.step)))

    def publish_networks():
        scheduler_learning_ready = bool(
            getattr(config, "aia_ablation", None) not in {"fixed_rule", "rl_only"}
            and len(scheduler_replay_buffer) >= scheduler_dqn_config.warmup_transitions
        )
        server.publish_network(
            {
                "rl_params": agent.state.params,
                "scheduler_params": scheduler_agent.state.params,
                # New actors may schedule immediately with the initialized Q
                # network. Keep scheduler_ready as the legacy learning-ready
                # field for compatibility with older actors.
                "scheduler_policy_ready": getattr(config, "aia_ablation", None)
                not in {"fixed_rule", "rl_only"},
                "scheduler_learning_ready": scheduler_learning_ready,
                "scheduler_ready": scheduler_learning_ready,
                "scheduler_step": scheduler_step_value(),
                "rl_policy_version": int(step),
            }
        )

    # Publish immediately so actor can start later without blocking learner startup.
    publish_networks()
    print_green("sent initial RL/Scheduler networks to actor")

    # Train from demos immediately. Once the actor has produced enough online data,
    # switch automatically to RLPD-style demo/online mixed batches.
    demo_frac = 0.5
    demo_bs = int(round(config.batch_size * demo_frac))
    demo_bs = max(1, min(config.batch_size - 1, demo_bs))
    online_bs = config.batch_size - demo_bs
    print_green(
        f"Initial sampling ratio: demo={config.batch_size}/{config.batch_size} (1.00), online=0/{config.batch_size} (0.00)"
    )
    print_green(
        f"Will switch to mixed sampling after online replay reaches {config.training_starts} transitions: "
        f"demo={demo_bs}/{config.batch_size}, online={online_bs}/{config.batch_size}"
    )

    replay_iterator = None
    using_online_replay = False

    demo_only_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": demo_bs,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    def next_train_batch():
        nonlocal replay_iterator, using_online_replay
        online_ready = len(replay_buffer) >= config.training_starts and online_bs > 0
        if online_ready:
            if replay_iterator is None:
                replay_iterator = replay_buffer.get_iterator(
                    sample_args={
                        "batch_size": online_bs,
                        "pack_obs_and_next_obs": True,
                    },
                    device=sharding.replicate(),
                )
            if not using_online_replay:
                using_online_replay = True
                print_green(
                    f"Online replay ready: {len(replay_buffer)} transitions. Switching to mixed demo/online batches."
                )
            return concat_batches(next(replay_iterator), next(demo_iterator), axis=0)
        return next(demo_only_iterator)

    scheduler_iterator = None
    scheduler_update_budget = 0
    last_scheduler_data_id = scheduler_replay_buffer.latest_data_id()

    def update_scheduler_for_new_options():
        """Spend update budget generated only by newly received Options."""
        nonlocal scheduler_agent
        nonlocal scheduler_iterator
        nonlocal scheduler_update_budget
        nonlocal last_scheduler_data_id

        latest_data_id = scheduler_replay_buffer.latest_data_id()
        new_options = max(0, latest_data_id - last_scheduler_data_id)
        last_scheduler_data_id = latest_data_id
        if getattr(config, "aia_ablation", None) in {"fixed_rule", "rl_only"}:
            return None, 0, new_options
        ready = len(scheduler_replay_buffer) >= scheduler_dqn_config.warmup_transitions
        if ready and new_options:
            scheduler_update_budget += (
                new_options * scheduler_dqn_config.updates_per_transition
            )
        if not ready or scheduler_update_budget <= 0:
            return None, 0, new_options

        if scheduler_iterator is None:
            scheduler_iterator = scheduler_replay_buffer.get_iterator(
                sample_args={
                    "batch_size": scheduler_dqn_config.batch_size,
                    "recent_fraction": (scheduler_dqn_config.recent_sample_fraction),
                    "recent_window": scheduler_dqn_config.recent_sample_window,
                },
                device=sharding.replicate(),
            )
        updates = min(
            scheduler_update_budget,
            scheduler_dqn_config.max_updates_per_loop,
        )
        latest_info = None
        for _ in range(updates):
            scheduler_batch = next(scheduler_iterator)
            scheduler_agent, latest_info = scheduler_agent.update(scheduler_batch)
        scheduler_update_budget -= updates
        return latest_info, updates, new_options

    # replay_iterator = replay_buffer.get_iterator(
    #     sample_args={
    #         "batch_size": config.batch_size // 2,
    #         "pack_obs_and_next_obs": True,
    #     },
    #     device=sharding.replicate(),
    # )
    # demo_iterator = demo_buffer.get_iterator(
    #     sample_args={
    #         "batch_size": config.batch_size // 2,
    #         "pack_obs_and_next_obs": True,
    #     },
    #     device=sharding.replicate(),
    # )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    last_scheduler_info = None
    last_scheduler_info_learner_step = -1

    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset(
            {"critic", "grasp_critic", "actor", "temperature"}
        )

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_train_batch"):
                batch = next_train_batch()

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next_train_batch()
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )

        with timer.context("train_scheduler"):
            (
                scheduler_info,
                scheduler_updates,
                new_scheduler_options,
            ) = update_scheduler_for_new_options()
            if scheduler_info is not None:
                last_scheduler_info = scheduler_info
                last_scheduler_info_learner_step = step

        # publish the updated network
        if scheduler_updates > 0 or (step > 0 and step % config.steps_per_update == 0):
            agent = jax.block_until_ready(agent)
            scheduler_agent = jax.block_until_ready(scheduler_agent)
            publish_networks()

        if step % config.log_period == 0 and wandb_logger:
            replay_intervention_stats = replay_buffer.get_intervention_stats()
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)
            scheduler_size = len(scheduler_replay_buffer)
            scheduler_metrics = {
                "scheduler/buffer_size": scheduler_size,
                "scheduler/policy_ready": int(
                    getattr(config, "aia_ablation", None)
                    not in {"fixed_rule", "rl_only"}
                ),
                "scheduler/learning_ready": int(
                    getattr(config, "aia_ablation", None)
                    not in {"fixed_rule", "rl_only"}
                    and scheduler_size >= scheduler_dqn_config.warmup_transitions
                ),
                "scheduler/ready": int(
                    getattr(config, "aia_ablation", None)
                    not in {"fixed_rule", "rl_only"}
                    and scheduler_size >= scheduler_dqn_config.warmup_transitions
                ),
                "scheduler/learning_starts_transitions": (
                    scheduler_dqn_config.warmup_transitions
                ),
                "scheduler/warmup_transitions": (
                    scheduler_dqn_config.warmup_transitions
                ),
                "scheduler/update_step": scheduler_step_value(),
                "scheduler/updates_this_loop": scheduler_updates,
                "scheduler/pending_update_budget": scheduler_update_budget,
                "scheduler/new_options_this_loop": new_scheduler_options,
                "scheduler/last_update_age": (
                    -1
                    if last_scheduler_info_learner_step < 0
                    else step - last_scheduler_info_learner_step
                ),
                "scheduler/epsilon": (
                    1.0
                    if getattr(config, "aia_ablation", None) == "fixed_rule"
                    else scheduler_dqn_config.epsilon(scheduler_step_value())
                ),
                "scheduler/learning_rate": scheduler_dqn_config.learning_rate,
                "scheduler/batch_size": scheduler_dqn_config.batch_size,
                "scheduler/updates_per_transition": (
                    scheduler_dqn_config.updates_per_transition
                ),
                "scheduler/max_updates_per_loop": (
                    scheduler_dqn_config.max_updates_per_loop
                ),
                "scheduler/target_update_tau": (scheduler_dqn_config.target_update_tau),
                "scheduler/gradient_clip": scheduler_dqn_config.gradient_clip,
                "scheduler/recent_sample_fraction": (
                    scheduler_dqn_config.recent_sample_fraction
                ),
                "scheduler/recent_sample_window": (
                    scheduler_dqn_config.recent_sample_window
                ),
            }
            if scheduler_size:
                scheduler_actions = scheduler_replay_buffer.dataset_dict["actions"][
                    :scheduler_size
                ]
                for option_id in OptionID:
                    scheduler_metrics[
                        f"scheduler/action_fraction/{option_id.name.lower()}"
                    ] = float(np.mean(scheduler_actions == int(option_id)))
                scheduler_metrics["scheduler/rl_policy_version_mean"] = float(
                    np.mean(
                        scheduler_replay_buffer.dataset_dict["rl_policy_version"][
                            :scheduler_size
                        ]
                    )
                )
            if last_scheduler_info is not None:
                scheduler_metrics.update(
                    {
                        f"scheduler/{key}": value
                        for key, value in last_scheduler_info.items()
                    }
                )
            wandb_logger.log(scheduler_metrics, step=step)
            wandb_logger.log(
                {
                    "buffer/online_size": len(replay_buffer),
                    "buffer/demo_size": len(demo_buffer),
                    "buffer/online_ready": int(
                        len(replay_buffer) >= config.training_starts
                    ),
                    "buffer/using_online_replay": int(using_online_replay),
                    "buffer/training_starts": config.training_starts,
                    "buffer/replay_total_samples": replay_intervention_stats[
                        "total_samples"
                    ],
                    "buffer/replay_human_intervention_samples": (
                        replay_intervention_stats["human_intervention_samples"]
                    ),
                    "buffer/replay_human_intervention_ratio": (
                        replay_intervention_stats["human_intervention_ratio"]
                    ),
                    "buffer/replay_other_strategy_intervention_samples": (
                        replay_intervention_stats["other_strategy_intervention_samples"]
                    ),
                    "buffer/replay_other_strategy_intervention_ratio": (
                        replay_intervention_stats["other_strategy_intervention_ratio"]
                    ),
                    "buffer/human_intervention_rate": replay_intervention_stats[
                        "human_intervention_ratio"
                    ],
                    "buffer/other_strategy_intervention_rate": replay_intervention_stats[
                        "other_strategy_intervention_ratio"
                    ],
                    "buffer/intervention_rate": replay_intervention_stats[
                        "intervention_ratio"
                    ],
                },
                step=step,
            )

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )
            scheduler_checkpoint_path = os.path.join(
                os.path.abspath(FLAGS.checkpoint_path), "scheduler_checkpoints"
            )
            checkpoints.save_checkpoint(
                scheduler_checkpoint_path,
                scheduler_agent.state,
                # Use the outer learner step for unique filenames even when no
                # new Option transition arrived between two checkpoints.
                step=step,
                prefix="scheduler_",
                keep=100,
            )


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    ablation = apply_ablation(
        config,
        FLAGS.exp_name,
        os.getenv("AIA_ABLATION", ""),
        os.getenv("SCHEDULER_DQN_EXPLORATION_WEIGHTS"),
    )
    if ablation is not None:
        if os.getenv("AIA_SEED") is not None:
            FLAGS.seed = int(os.environ["AIA_SEED"])
        if FLAGS.seed < 0:
            raise ValueError("AIA_SEED must be nonnegative")
        ablation["seed"] = FLAGS.seed
        # Keep the Option/Probe lifecycle shared; fixed_rule bypasses only DQN.
        FLAGS.learned_option_scheduler = ablation["group"] != "rl_only"
        FLAGS.manual_option_scheduler = False
        print("[AIA ablation]", ablation, flush=True)
    check_manifest(FLAGS.checkpoint_path, ablation, resume=FLAGS.resume_training)

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)

    if (
        config.setup_mode == "single-arm-fixed-gripper"
        or config.setup_mode == "dual-arm-fixed-gripper"
    ):
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == "single-arm-learned-gripper":
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == "dual-arm-learned-gripper":
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree.map(jnp.array, agent), sharding.replicate())

    if FLAGS.resume_training:
        if FLAGS.checkpoint_path is None or not os.path.exists(FLAGS.checkpoint_path):
            raise FileNotFoundError(
                "--resume_training requires an existing --checkpoint_path. "
                "Start a new run by omitting --resume_training or choose the saved run path."
            )
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        latest_ckpt = checkpoints.latest_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path)
        )
        ckpt_step = latest_step_from_path(latest_ckpt, "checkpoint_")
        print_green(f"Loaded previous checkpoint at step {ckpt_step}.")
    elif (
        FLAGS.checkpoint_path is not None
        and os.path.exists(FLAGS.checkpoint_path)
        and not FLAGS.allow_existing_checkpoint_path
    ):
        latest_ckpt = checkpoints.latest_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path)
        )
        buffer_files = glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer", "*.pkl"))
        demo_buffer_files = glob.glob(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer", "*.pkl")
        )
        scheduler_buffer_files = glob.glob(
            os.path.join(FLAGS.checkpoint_path, "scheduler_buffer", "*.pkl")
        )
        scheduler_checkpoint_files = glob.glob(
            os.path.join(FLAGS.checkpoint_path, "scheduler_checkpoints", "scheduler_*")
        )
        if (
            latest_ckpt
            or buffer_files
            or demo_buffer_files
            or scheduler_buffer_files
            or scheduler_checkpoint_files
        ):
            raise FileExistsError(
                f"Checkpoint path already has training state: {FLAGS.checkpoint_path}. "
                "Use --resume_training to continue it, or use a new --checkpoint_path."
            )

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = InterventionTrackingReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        wandb_suffix = str(os.environ.get("WANDB_DESCRIPTOR_SUFFIX", "")).strip()
        wandb_description = (
            f"{FLAGS.exp_name}_{os.path.basename(os.path.abspath(FLAGS.checkpoint_path))}"
            if FLAGS.checkpoint_path is not None
            else FLAGS.exp_name
        )
        if wandb_suffix:
            wandb_description = f"{FLAGS.exp_name}_{wandb_suffix}_{os.path.basename(os.path.abspath(FLAGS.checkpoint_path))}"
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="recover",
            entity="VLA-data",
            description=wandb_description,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    scheduler_state_dim = (
        int(getattr(config, "scheduler_expected_rl_feature_dim", 576))
        + int(getattr(config, "scheduler_history_length", 4))
        * (OPTION_HISTORY_ITEM_DIM + 1)
        + policy_change_probe_state_feature_dim(config)
    )
    scheduler_replay_capacity = int(
        getattr(config, "scheduler_replay_buffer_capacity", 50_000)
    )
    scheduler_dqn_defaults = SchedulerDQNConfig()
    scheduler_dqn_config = SchedulerDQNConfig(
        hidden_dims=tuple(
            getattr(
                config,
                "scheduler_dqn_hidden_dims",
                scheduler_dqn_defaults.hidden_dims,
            )
        ),
        learning_rate=float(
            getattr(
                config,
                "scheduler_dqn_learning_rate",
                scheduler_dqn_defaults.learning_rate,
            )
        ),
        batch_size=int(
            getattr(
                config,
                "scheduler_dqn_batch_size",
                scheduler_dqn_defaults.batch_size,
            )
        ),
        warmup_transitions=int(
            getattr(
                config,
                "scheduler_dqn_warmup_transitions",
                scheduler_dqn_defaults.warmup_transitions,
            )
        ),
        updates_per_transition=int(
            getattr(
                config,
                "scheduler_dqn_updates_per_transition",
                scheduler_dqn_defaults.updates_per_transition,
            )
        ),
        max_updates_per_loop=int(
            getattr(
                config,
                "scheduler_dqn_max_updates_per_loop",
                scheduler_dqn_defaults.max_updates_per_loop,
            )
        ),
        recent_sample_fraction=float(
            getattr(
                config,
                "scheduler_dqn_recent_sample_fraction",
                scheduler_dqn_defaults.recent_sample_fraction,
            )
        ),
        recent_sample_window=int(
            getattr(
                config,
                "scheduler_dqn_recent_sample_window",
                scheduler_dqn_defaults.recent_sample_window,
            )
        ),
        target_update_tau=float(
            getattr(
                config,
                "scheduler_dqn_target_update_tau",
                scheduler_dqn_defaults.target_update_tau,
            )
        ),
        gradient_clip=float(
            getattr(
                config,
                "scheduler_dqn_gradient_clip",
                scheduler_dqn_defaults.gradient_clip,
            )
        ),
        epsilon_start=float(
            getattr(
                config,
                "scheduler_dqn_epsilon_start",
                scheduler_dqn_defaults.epsilon_start,
            )
        ),
        epsilon_end=float(
            getattr(
                config,
                "scheduler_dqn_epsilon_end",
                scheduler_dqn_defaults.epsilon_end,
            )
        ),
        epsilon_decay_steps=int(
            getattr(
                config,
                "scheduler_dqn_epsilon_decay_steps",
                scheduler_dqn_defaults.epsilon_decay_steps,
            )
        ),
        exploration_weights=(
            None
            if getattr(
                config,
                "scheduler_dqn_exploration_weights",
                scheduler_dqn_defaults.exploration_weights,
            )
            is None
            else tuple(
                float(weight)
                for weight in getattr(
                    config,
                    "scheduler_dqn_exploration_weights",
                    scheduler_dqn_defaults.exploration_weights,
                )
            )
        ),
    )
    rng, scheduler_rng = jax.random.split(rng)
    scheduler_agent = SchedulerDQNAgent.create(
        scheduler_rng,
        state_dim=scheduler_state_dim,
        num_options=len(OptionID),
        config=scheduler_dqn_config,
    )
    scheduler_agent = jax.device_put(
        jax.tree.map(jnp.array, scheduler_agent), sharding.replicate()
    )

    if FLAGS.resume_training and FLAGS.checkpoint_path is not None:
        scheduler_checkpoint_path = os.path.join(
            os.path.abspath(FLAGS.checkpoint_path), "scheduler_checkpoints"
        )
        latest_scheduler_ckpt = checkpoints.latest_checkpoint(
            scheduler_checkpoint_path, prefix="scheduler_"
        )
        if latest_scheduler_ckpt is not None:
            scheduler_state = checkpoints.restore_checkpoint(
                scheduler_checkpoint_path,
                scheduler_agent.state,
                prefix="scheduler_",
            )
            scheduler_agent = scheduler_agent.replace(state=scheduler_state)
            print_green(
                "Loaded Scheduler checkpoint at update step "
                f"{int(np.asarray(jax.device_get(scheduler_agent.state.step)))}."
            )
        else:
            print_green(
                "No Scheduler checkpoint found; starting Scheduler from scratch."
            )

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        scheduler_replay_buffer = SchedulerReplayBufferDataStore(
            state_dim=scheduler_state_dim,
            num_options=len(OptionID),
            capacity=scheduler_replay_capacity,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            print("Demo path is: ", path, "Current working directory is: ", os.getcwd())
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    transition = prepare_loaded_transition(
                        transition, env, include_grasp_penalty
                    )
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")
        print_green(
            f"scheduler buffer size: {len(scheduler_replay_buffer)} "
            f"state_dim={scheduler_state_dim}"
        )

        if (
            FLAGS.resume_training
            and FLAGS.checkpoint_path is not None
            and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer"))
        ):
            for file in natsorted(
                glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
            ):
                try:
                    transitions = load_transition_file(file)
                except Exception as exc:
                    print(f"Skipping unreadable buffer file {file}: {exc}")
                    continue
                for transition in transitions:
                    transition = prepare_loaded_transition(
                        transition, env, include_grasp_penalty
                    )
                    replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if (
            FLAGS.resume_training
            and FLAGS.checkpoint_path is not None
            and os.path.exists(os.path.join(FLAGS.checkpoint_path, "demo_buffer"))
        ):
            for file in natsorted(
                glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl"))
            ):
                try:
                    transitions = load_transition_file(file)
                except Exception as exc:
                    print(f"Skipping unreadable demo_buffer file {file}: {exc}")
                    continue
                for transition in transitions:
                    transition = prepare_loaded_transition(
                        transition, env, include_grasp_penalty
                    )
                    demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        if (
            FLAGS.resume_training
            and FLAGS.checkpoint_path is not None
            and os.path.exists(os.path.join(FLAGS.checkpoint_path, "scheduler_buffer"))
        ):
            for file in natsorted(
                glob.glob(
                    os.path.join(
                        FLAGS.checkpoint_path,
                        "scheduler_buffer/scheduler_transitions_*.pkl",
                    )
                )
            ):
                try:
                    transitions = load_scheduler_transition_file(file)
                except Exception as exc:
                    print(f"Skipping unreadable scheduler buffer file {file}: {exc}")
                    continue
                for transition in transitions:
                    try:
                        scheduler_replay_buffer.insert(transition)
                    except Exception as exc:
                        print(
                            f"Skipping incompatible scheduler transition in {file}: {exc}"
                        )
            print_green(
                "Loaded previous Scheduler data. "
                f"Scheduler replay buffer size: {len(scheduler_replay_buffer)}"
            )

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            scheduler_agent,
            scheduler_dqn_config,
            replay_buffer,
            demo_buffer=demo_buffer,
            scheduler_replay_buffer=scheduler_replay_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)
        scheduler_data_store = QueuedDataStore(scheduler_replay_capacity)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            scheduler_agent,
            scheduler_dqn_config,
            data_store,
            intvn_data_store,
            scheduler_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    print(os.getcwd())
    app.run(main)
