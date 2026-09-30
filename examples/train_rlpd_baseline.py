#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import json
import pickle as pkl
import threading
import select
import sys
import termios
import tty
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING
import math

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name", "usb_insert", "Name of experiment corresponding to folder."
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
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging
flags.DEFINE_boolean(
    "baseline_recovery_enabled",
    False,
    "Enable local InterveneNet + RecoveryNet action override on the actor.",
)
flags.DEFINE_string(
    "baseline_model_root",
    "",
    "Repository root containing baselines/farl_inspired for the local recovery heads.",
)
flags.DEFINE_string(
    "baseline_intervene_ckpt_path",
    "",
    "Path to InterveneNet model.pt.",
)
flags.DEFINE_string(
    "baseline_recovery_ckpt_path",
    "",
    "Path to RecoveryNet model.pt.",
)
flags.DEFINE_string(
    "baseline_recovery_device",
    "cuda",
    "Torch device for the local baseline recovery heads.",
)
flags.DEFINE_string(
    "baseline_recovery_image_keys",
    "external,wrist",
    "Comma-separated observation image keys ordered as side/external camera, then wrist camera.",
)
flags.DEFINE_float(
    "baseline_intervene_threshold",
    0.5,
    "tau_on: enter RecoveryNet override when p_intervene >= this value.",
)
flags.DEFINE_float(
    "baseline_recovery_off_threshold",
    0.4,
    "tau_off: count one recovery-exit vote when p_intervene < this value.",
)
flags.DEFINE_integer(
    "baseline_recovery_exit_low_steps",
    2,
    "Exit RecoveryNet override after this many consecutive p_intervene < tau_off steps.",
)
flags.DEFINE_integer(
    "baseline_recovery_max_steps",
    20,
    "Maximum consecutive RecoveryNet override steps per trigger.",
)
flags.DEFINE_string(
    "baseline_recovery_log_dir",
    "",
    "Optional directory for JSONL logs of InterveneNet/RecoveryNet inputs and outputs.",
)
flags.DEFINE_integer(
    "baseline_recovery_log_every_steps",
    1,
    "Log every N actor steps while baseline recovery logging is enabled.",
)
flags.DEFINE_boolean(
    "baseline_recovery_log_only_active",
    False,
    "Only log steps where RecoveryNet is active or just triggered.",
)
flags.DEFINE_boolean(
    "baseline_recovery_log_images",
    False,
    "Also save raw image inputs as compressed npz files. This can be large.",
)
flags.DEFINE_string(
    "baseline_recovery_server_python",
    "",
    "Deprecated; baseline_recovery_server.py must be started manually.",
)
flags.DEFINE_integer(
    "baseline_recovery_zmq_port",
    15567,
    "TCP port for ZMQ REQ/REP between actor and baseline_recovery_server.",
)
flags.DEFINE_boolean(
    "baseline_recovery_server_external",
    False,
    "Required when baseline recovery is enabled; connect to an already-running server.",
)


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


class ManualEpisodeLabeler:
    def __init__(self):
        self.success = False
        self.failure = False
        self.baseline_toggle = False
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

        print(
            "Manual actor labels: SPACE/s = success and reset, "
            "f/ESC = failure and reset, q = toggle baseline recovery inference."
        )

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
        return None

    def clear(self):
        self._poll_stdin()
        self.success = False
        self.failure = False

    def consume_baseline_toggle(self):
        self._poll_stdin()
        if self.baseline_toggle:
            self.baseline_toggle = False
            return True
        return False

    def _mark_char(self, char):
        if char in (" ", "s", "S"):
            self.success = True
        elif char in ("f", "F", "\x1b"):
            self.failure = True
        elif char in ("q", "Q"):
            self.baseline_toggle = True

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


def _peek_spacemouse_intervention(env):
    for current in _iter_env_chain(env):
        method = getattr(current, "peek_intervention_action", None)
        if not callable(method):
            continue
        action = method()
        if action is not None:
            return np.asarray(action, dtype=np.float32)
    return None


def _get_env_action_scale(env):
    for current in _iter_env_chain(env):
        scale = getattr(current, "action_scale", None)
        if scale is None:
            continue
        arr = np.asarray(scale, dtype=np.float32).reshape(-1)
        if arr.size >= 3:
            return arr[:3]
    raise RuntimeError(
        "baseline recovery requires env.action_scale with at least 3 values"
    )


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


def load_transition_file(path):
    try:
        with open(path, "rb") as f:
            obj = pkl.load(f)
    except EOFError:
        with open(path, "rb") as f:
            data = f.read()
        obj = pkl.loads(data + b"ue.")
        print(f"Recovered truncated pickle while loading {path}")
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


class BaselineRecoveryController:
    def __init__(self, action_space, *, action_scale=None, gripper_deadband=0.05):
        self.enabled = bool(FLAGS.baseline_recovery_enabled)
        self.runtime_enabled = bool(self.enabled)
        self.action_space = action_space
        self.action_scale = np.asarray(
            action_scale if action_scale is not None else [0.05, 0.1, 1.0],
            dtype=np.float32,
        ).reshape(-1)
        if self.action_scale.size < 3:
            self.action_scale = np.pad(
                self.action_scale, (0, 3 - self.action_scale.size), constant_values=1.0
            )
        self.gripper_deadband = float(gripper_deadband)
        self.active = False
        self.recovery_steps = 0
        self.low_streak = 0
        self.last_action = None
        self.recovery_queue = []
        self.recovery_chunk_total = 0
        if not self.enabled:
            return

        import zmq

        self.image_keys = [
            key.strip()
            for key in str(FLAGS.baseline_recovery_image_keys).split(",")
            if key.strip()
        ]
        self.log_dir = str(FLAGS.baseline_recovery_log_dir or "").strip()
        self.log_file = None
        self.log_index = 0
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_file = os.path.join(
                self.log_dir,
                f"baseline_recovery_actor_{int(time.time())}_{os.getpid()}.jsonl",
            )

        port = int(FLAGS.baseline_recovery_zmq_port)

        self._zmq_server_proc = None
        if not FLAGS.baseline_recovery_server_external:
            raise RuntimeError(
                "baseline recovery no longer auto-starts baseline_recovery_server.py. "
                "Start the server manually, then pass --baseline_recovery_server_external "
                f"and --baseline_recovery_zmq_port={port} to the actor."
            )
        print(
            "[baseline_recovery] connecting to external server on port",
            port,
            flush=True,
        )

        self._zmq_ctx = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.REQ)
        self._zmq_sock.connect(f"tcp://127.0.0.1:{port}")

        # Wait for server to be ready (up to 60 s)
        info = None
        for _ in range(60):
            time.sleep(1.0)
            try:
                self._zmq_sock.send(pkl.dumps({"cmd": "ping"}), flags=zmq.NOBLOCK)
                if self._zmq_sock.poll(timeout=2000):
                    info = pkl.loads(self._zmq_sock.recv())
                    if info.get("status") == "ok":
                        break
            except Exception:
                # Reset socket state on send/recv error
                self._zmq_sock.close()
                self._zmq_sock = self._zmq_ctx.socket(zmq.REQ)
                self._zmq_sock.connect(f"tcp://127.0.0.1:{port}")
        else:
            raise RuntimeError(
                "baseline_recovery_server did not become ready within 60 s"
            )

        self.action_dim = int(
            info.get("state_dim", info.get("target_dim", info.get("action_dim", 7)))
        )
        self.intervene_num_cam = int(info.get("intervene_num_cam", 2))
        self.intervene_image_size = int(info.get("intervene_image_size", 224))
        self.recovery_num_cam = int(info.get("recovery_num_cam", 2))
        self.recovery_image_size = int(info.get("recovery_image_size", 224))

        print_green(
            "baseline_recovery_server ready (ZMQ): "
            f"intervene={FLAGS.baseline_intervene_ckpt_path}, "
            f"recovery={FLAGS.baseline_recovery_ckpt_path}, "
            f"device={info.get('device')}, image_keys={self.image_keys}"
        )

    def reset(self):
        self.active = False
        self.recovery_steps = 0
        self.low_streak = 0
        self.last_action = None
        self.recovery_queue = []
        self.recovery_chunk_total = 0

    def clear_recovery_queue(self):
        dropped = len(self.recovery_queue)
        self.active = False
        self.recovery_steps = 0
        self.low_streak = 0
        self.recovery_queue = []
        self.recovery_chunk_total = 0
        return dropped

    def toggle_runtime(self):
        if not self.enabled:
            print(
                "[baseline_recovery] q pressed, but baseline recovery is disabled "
                "because --baseline_recovery_enabled=false.",
                flush=True,
            )
            return False

        dropped = self.clear_recovery_queue()
        self.runtime_enabled = not self.runtime_enabled
        state = "ENABLED" if self.runtime_enabled else "DISABLED"
        print(
            "\n"
            "============================================================\n"
            f"[BASELINE RECOVERY {state}] q pressed; "
            f"extra InterveneNet/RecoveryNet inference is now {state.lower()}.\n"
            f"Cleared pending recovery chunk actions: {dropped}.\n"
            "============================================================\n",
            flush=True,
        )
        return self.runtime_enabled

    def __del__(self):
        try:
            self._zmq_sock.close()
            self._zmq_ctx.term()
        except Exception:
            pass
        try:
            if self._zmq_server_proc is not None:
                self._zmq_server_proc.terminate()
        except Exception:
            pass

    def _zmq_call(self, request: dict) -> dict:
        self._zmq_sock.send(pkl.dumps(request))
        return pkl.loads(self._zmq_sock.recv())

    def _extract_obs_images(self, obs) -> dict:
        if not isinstance(obs, dict):
            return {}
        return {k: np.asarray(obs[k]) for k in self.image_keys if k in obs}

    def _obs_image_summary(self, obs):
        out = {}
        if not isinstance(obs, dict):
            return out
        for key in self.image_keys:
            if key not in obs:
                continue
            arr = np.asarray(obs[key])
            finite = (
                arr[np.isfinite(arr)]
                if np.issubdtype(arr.dtype, np.number)
                else np.asarray([])
            )
            out[key] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": float(np.min(finite)) if finite.size else None,
                "max": float(np.max(finite)) if finite.size else None,
            }
        return out

    def _save_image_inputs(self, obs, actor_step):
        if not bool(FLAGS.baseline_recovery_log_images) or not self.log_dir:
            return None
        arrays = {}
        if isinstance(obs, dict):
            for key in self.image_keys:
                if key in obs:
                    arrays[key.replace("/", "_")] = np.asarray(obs[key])
        if not arrays:
            return None
        step_tag = "none" if actor_step is None else str(int(actor_step))
        path = os.path.join(
            self.log_dir,
            f"baseline_recovery_inputs_step{step_tag}_{self.log_index:06d}.npz",
        )
        np.savez_compressed(path, **arrays)
        return path

    def _single_state_payload(self, state):
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        payload = {"state": arr.round(6).tolist()}
        if arr.size >= 7:
            payload["gripper_state"] = arr[0:1].round(6).tolist()
            payload["tcp_pose"] = arr[1:7].round(6).tolist()
        elif arr.size >= 6:
            payload["tcp_pose"] = arr[-6:].round(6).tolist()
        return payload

    def _state_payload(self, state):
        if state is None:
            return None
        arr = np.asarray(state, dtype=np.float32)
        if arr.size == 0:
            return {"state": []}
        if arr.ndim <= 1:
            return self._single_state_payload(arr)
        arr = arr.reshape(-1, arr.shape[-1])
        return [self._single_state_payload(row) for row in arr]

    def _log_record(self, record):
        if not self.log_dir:
            return
        episode_id = record.get("episode_id")
        if episode_id is not None:
            path = os.path.join(self.log_dir, f"episode_{episode_id:06d}.jsonl")
        else:
            path = self.log_file
        if not path:
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _obs_to_recovery_state(self, obs):
        if not isinstance(obs, dict) or "state" not in obs:
            return None
        st = np.asarray(obs["state"], dtype=np.float32)
        if st.ndim >= 2:
            st = st.reshape(-1, st.shape[-1])[-1]
        st = np.asarray(st, dtype=np.float32).reshape(-1)

        if st.size == self.action_dim:
            return st.astype(np.float32, copy=False)
        if st.size == 11 and self.action_dim == 7:
            return np.concatenate([st[0:1], st[5:11]], axis=0).astype(
                np.float32, copy=False
            )
        if st.size == 9 and self.action_dim == 7:
            st11 = np.zeros((11,), dtype=np.float32)
            st11[2:] = st
            return np.concatenate([st11[0:1], st11[5:11]], axis=0).astype(
                np.float32, copy=False
            )

        if st.size > self.action_dim:
            return st[: self.action_dim].astype(np.float32, copy=False)
        return np.pad(st, (0, self.action_dim - st.size), constant_values=0.0).astype(
            np.float32
        )

    def _to_env_action(self, action):
        env_shape = tuple(self.action_space.shape)
        env_dim = int(np.prod(env_shape))
        flat = np.asarray(action, dtype=np.float32).reshape(-1)
        if flat.size > env_dim:
            flat = flat[:env_dim]
        elif flat.size < env_dim:
            flat = np.pad(flat, (0, env_dim - flat.size), constant_values=0.0)
        return clip_action_to_space(flat.reshape(env_shape), self.action_space)

    def _normalize_state_chunk(self, states):
        arr = np.asarray(states, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, int(getattr(self, "action_dim", 7))), dtype=np.float32)

        state_dim = int(
            getattr(self, "action_dim", int(np.prod(self.action_space.shape)))
        )
        if arr.ndim <= 1:
            flat = arr.reshape(-1)
            if flat.size > state_dim and flat.size % state_dim == 0:
                arr = flat.reshape(-1, state_dim)
            else:
                arr = flat.reshape(1, -1)
        else:
            arr = arr.reshape(-1, arr.shape[-1])

        if arr.shape[1] > state_dim:
            arr = arr[:, :state_dim]
        elif arr.shape[1] < state_dim:
            arr = np.pad(
                arr, ((0, 0), (0, state_dim - arr.shape[1])), constant_values=0.0
            )
        return arr.astype(np.float32, copy=False)

    def _state_chunk_to_target_queue(self, states):
        arr = self._normalize_state_chunk(states)
        return [row.copy() for row in arr]

    def _target_state_to_env_delta_action(self, current_state, target_state):
        cur = self._normalize_state_chunk(current_state)
        tgt = self._normalize_state_chunk(target_state)
        if cur.shape[0] == 0 or tgt.shape[0] == 0:
            return None
        cur_state = cur[-1]
        next_state = tgt[-1]
        xyz_scale = max(float(self.action_scale[0]), 1e-6)
        rot_scale = max(float(self.action_scale[1]), 1e-6)
        action = np.zeros(
            (max(7, int(np.prod(self.action_space.shape))),), dtype=np.float32
        )
        action[:3] = np.clip((next_state[1:4] - cur_state[1:4]) / xyz_scale, -1.0, 1.0)
        action[3:6] = np.clip(
            (next_state[4:7] - cur_state[4:7]) / (rot_scale / 4.0), -1.0, 1.0
        )
        if next_state[0] > cur_state[0] + self.gripper_deadband:
            action[6] = 1.0
        elif next_state[0] < cur_state[0] - self.gripper_deadband:
            action[6] = -1.0
        return self._to_env_action(action)

    def select_action(self, obs, task_action, actor_step=None, episode_id=None):
        if not self.enabled:
            return task_action, {}
        if not self.runtime_enabled:
            return task_action, {
                "baseline_recovery_runtime_enabled": False,
                "baseline_recovery_active": False,
                "baseline_recovery_paused_active": bool(self.active),
                "baseline_recovery_steps": int(self.recovery_steps),
                "action_source": "policy",
            }

        try:
            self.log_index += 1
            log_every = max(1, int(FLAGS.baseline_recovery_log_every_steps))
            should_log = bool(self.log_file) and (
                actor_step is None or int(actor_step) % log_every == 0
            )
            task_action_for_log = np.asarray(task_action, dtype=np.float32).reshape(-1)
            recovery_input_state = None
            recovery_output_state = None
            just_triggered = False
            p_intervene = None

            if not self.recovery_queue:
                reply = self._zmq_call(
                    {"cmd": "intervene", "obs_images": self._extract_obs_images(obs)}
                )
                if "error" in reply:
                    raise RuntimeError(f"intervene server error: {reply['error']}")
                p_intervene = float(reply["p_intervene"])

                if p_intervene >= float(FLAGS.baseline_intervene_threshold):
                    current_state = self._obs_to_recovery_state(obs)
                    if current_state is None:
                        raise RuntimeError(
                            "missing obs.state for RecoveryNet current_state input"
                        )
                    recovery_input_state = current_state.copy()
                    reply = self._zmq_call(
                        {
                            "cmd": "recovery",
                            "obs_images": self._extract_obs_images(obs),
                            "state": current_state,
                            "current_state": current_state,
                        }
                    )
                    if "error" in reply:
                        raise RuntimeError(f"recovery server error: {reply['error']}")
                    recovery_state_value = reply.get(
                        "state",
                        reply.get("recovery_state", reply.get("recovery_action")),
                    )
                    if recovery_state_value is None:
                        raise RuntimeError("recovery server response missing state")
                    recovery_state = np.asarray(
                        recovery_state_value,
                        dtype=np.float32,
                    )
                    recovery_output_state = recovery_state.copy()
                    self.recovery_queue = self._state_chunk_to_target_queue(
                        recovery_state
                    )
                    self.recovery_chunk_total = len(self.recovery_queue)
                    self.recovery_steps = 0
                    self.low_streak = 0
                    just_triggered = bool(self.recovery_queue)
                    if just_triggered:
                        self.active = True
                        current_state_print = (
                            np.asarray(current_state).reshape(-1).round(6).tolist()
                        )
                        recovery_state_print = (
                            np.asarray(recovery_state).round(6).tolist()
                        )
                        first_action = self._target_state_to_env_delta_action(
                            current_state,
                            self.recovery_queue[0],
                        )
                        first_action_print = (
                            None
                            if first_action is None
                            else np.asarray(first_action).reshape(-1).round(6).tolist()
                        )
                        print(
                            "\n"
                            "============================================================\n"
                            f"[BASELINE RECOVERY TRIGGERED] p_intervene={p_intervene:.3f} "
                            f">= tau_on={float(FLAGS.baseline_intervene_threshold):.3f}\n"
                            f"current_state={current_state_print}\n"
                            f"recovery_target_state_chunk={recovery_state_print}\n"
                            f"first_delta_action_preview={first_action_print}\n"
                            f"Queued RecoveryNet target state chunk with {self.recovery_chunk_total} env steps.\n"
                            "InterveneNet/RecoveryNet will not run again until this chunk finishes.\n"
                            "============================================================\n",
                            flush=True,
                        )

            if self.recovery_queue:
                target_state = self.recovery_queue.pop(0)
                current_state = self._obs_to_recovery_state(obs)
                action = self._target_state_to_env_delta_action(
                    current_state, target_state
                )
                if action is None:
                    raise RuntimeError("missing obs.state for RecoveryNet delta action")
                self.recovery_steps += 1
                self.last_action = action
                action_source = "baseline_recovery"
                remaining = len(self.recovery_queue)
                print(
                    f"[baseline_recovery] chunk_step={self.recovery_steps}/"
                    f"{self.recovery_chunk_total} remaining={remaining} "
                    f"p_intervene={'n/a' if p_intervene is None else f'{p_intervene:.3f}'} "
                    f"current_state={np.asarray(current_state).reshape(-1).round(4).tolist()} "
                    f"target_state={np.asarray(target_state).reshape(-1).round(4).tolist()} "
                    f"action={np.asarray(action).reshape(-1).round(4).tolist()}",
                    flush=True,
                )
                if not self.recovery_queue:
                    self.active = False
                    print(
                        "[baseline_recovery] completed recovery action chunk; "
                        "returning to normal policy/intervention checks.",
                        flush=True,
                    )
            else:
                action = task_action
                self.last_action = task_action
                self.active = False
                self.recovery_steps = 0
                self.low_streak = 0
                self.recovery_chunk_total = 0
                action_source = "policy"

            if should_log:
                if bool(FLAGS.baseline_recovery_log_only_active) and not (
                    just_triggered or action_source == "baseline_recovery"
                ):
                    should_log = False
            if should_log:
                image_npz_path = self._save_image_inputs(obs, actor_step)
                record = {
                    "time": time.time(),
                    "episode_id": None if episode_id is None else int(episode_id),
                    "actor_step": None if actor_step is None else int(actor_step),
                    "image_keys": list(self.image_keys),
                    "image_summary": self._obs_image_summary(obs),
                    "task_action_input": task_action_for_log.tolist(),
                    "intervene_input": {
                        "num_cam": int(self.intervene_num_cam),
                        "image_size": int(self.intervene_image_size),
                    },
                    "intervene_output": {
                        "p_intervene": None
                        if p_intervene is None
                        else float(p_intervene),
                        "tau_on": float(FLAGS.baseline_intervene_threshold),
                        "tau_off": float(FLAGS.baseline_recovery_off_threshold),
                        "triggered": bool(just_triggered),
                    },
                    "recovery_input": {
                        "active": bool(action_source == "baseline_recovery"),
                        "num_cam": int(self.recovery_num_cam),
                        "image_size": int(self.recovery_image_size),
                        "state_dim": int(self.action_dim),
                        "state": (
                            None
                            if recovery_input_state is None
                            else np.asarray(recovery_input_state).reshape(-1).tolist()
                        ),
                        "state_payload": self._state_payload(recovery_input_state),
                    },
                    "recovery_output": {
                        "state_dim": int(self.action_dim),
                        "state": (
                            None
                            if recovery_output_state is None
                            else np.asarray(recovery_output_state).tolist()
                        ),
                        "state_payload": self._state_payload(recovery_output_state),
                    },
                    "final_action": np.asarray(action, dtype=np.float32)
                    .reshape(-1)
                    .tolist(),
                    "action_source": action_source,
                    "recovery_state": {
                        "active": bool(self.active),
                        "steps": int(self.recovery_steps),
                        "chunk_total": int(self.recovery_chunk_total),
                        "queue_remaining": int(len(self.recovery_queue)),
                        "low_streak": int(self.low_streak),
                    },
                    "image_npz_path": image_npz_path,
                }
                self._log_record(record)

            return action, {
                "baseline_intervene_prob": p_intervene,
                "baseline_recovery_runtime_enabled": True,
                "baseline_recovery_active": bool(action_source == "baseline_recovery"),
                "baseline_recovery_steps": int(self.recovery_steps),
                "baseline_recovery_chunk_total": int(self.recovery_chunk_total),
                "baseline_recovery_queue_remaining": int(len(self.recovery_queue)),
                "action_source": action_source,
            }
        except Exception as exc:
            self.enabled = False
            self.runtime_enabled = False
            self.reset()
            print(
                f"Baseline recovery disabled after inference error: {exc}",
                flush=True,
            )
            return task_action, {
                "baseline_recovery_error": str(exc),
                "action_source": "policy",
            }


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    action_filter = EMAActionFilter(hz=10, cutoff_hz=2)
    baseline_recovery = BaselineRecoveryController(
        env.action_space,
        action_scale=_get_env_action_scale(env),
    )
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
        eval_step = 0
        try:
            for episode in range(FLAGS.eval_n_trajs):
                obs, _ = env.reset()
                manual_labeler.clear()
                baseline_recovery.reset()
                done = False
                truncated = False
                start_time = time.time()
                while not (done or truncated):
                    if manual_labeler.consume_baseline_toggle():
                        baseline_recovery.toggle_runtime()

                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs), argmax=True, seed=key
                    )
                    actions = np.asarray(jax.device_get(actions))
                    spacemouse_action = _peek_spacemouse_intervention(env)
                    if spacemouse_action is not None:
                        actions = clip_action_to_space(
                            spacemouse_action, env.action_space
                        )
                        baseline_recovery.reset()
                        baseline_meta = {
                            "baseline_recovery_skipped_for_spacemouse": True,
                            "action_source": "spacemouse",
                        }
                    else:
                        actions, baseline_meta = baseline_recovery.select_action(
                            obs, actions, actor_step=eval_step, episode_id=episode
                        )
                        if baseline_meta.get("action_source", "policy") in (
                            "policy",
                            "baseline_recovery",
                        ):
                            actions = action_filter(actions)
                    next_obs, reward, done, truncated, info = env.step(actions)
                    eval_step += 1
                    info.update(baseline_meta)
                    if "intervene_action" in info:
                        actions = info.pop("intervene_action")
                        actions = clip_action_to_space(actions, env.action_space)
                        baseline_recovery.reset()
                        info["baseline_recovery_overridden_by_spacemouse"] = True
                        print(
                            "[BASELINE RECOVERY OVERRIDDEN] SpaceMouse intervention detected; "
                            "RecoveryNet state reset.",
                            flush=True,
                        )
                    else:
                        actions = info.pop("executed_action", actions)
                    obs = next_obs

                    manual_label = manual_labeler.consume()
                    if manual_label == "success":
                        baseline_recovery.reset()
                        reward = 1.0
                        done = True
                        truncated = False
                        print("manual eval success marked; resetting environment")
                    elif manual_label == "failure":
                        baseline_recovery.reset()
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
                        baseline_recovery.reset()
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
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=500,
    )

    pending_network = {"params": None, "count": 0, "applied": 0}
    pending_network_lock = threading.Lock()

    def update_params(params):
        with pending_network_lock:
            pending_network["params"] = params
            pending_network["count"] += 1

    def apply_pending_network(force=False):
        nonlocal agent
        if not force and step % config.steps_per_update != 0:
            return
        with pending_network_lock:
            params = pending_network["params"]
            count = pending_network["count"]
            pending_network["params"] = None
        if params is not None:
            agent = agent.replace(state=agent.state.replace(params=params))
            skipped = max(0, count - pending_network["applied"] - 1)
            pending_network["applied"] = count
            if skipped:
                print_green(
                    f"Applied latest learner params; skipped {skipped} stale updates"
                )
            else:
                print_green("Applied latest learner params")

    client.recv_network_callback(update_params)

    current_trajectory = []
    trajectory_index = 0
    actor_stats = {
        "saved_trajectories": 0,
        "saved_success_trajectories": 0,
        "saved_failure_trajectories": 0,
        "saved_intervention_trajectories": 0,
        "saved_policy_trajectories": 0,
        "saved_transitions": 0,
        "saved_intervention_samples": 0,
    }
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
        transition["infos"] = info
        return transition

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
        if intervention_samples:
            demo_filename = f"interventions_{step}_{trajectory_index:06d}_{label}.pkl"
            demo_path = os.path.join(
                FLAGS.checkpoint_path, "demo_buffer", demo_filename
            )
            atomic_pickle_dump(intervention_samples, demo_path)
        actor_stats["saved_trajectories"] += 1
        actor_stats["saved_success_trajectories"] += int(success)
        actor_stats["saved_failure_trajectories"] += int(not success)
        actor_stats["saved_intervention_trajectories"] += int(trajectory_had_intvn)
        actor_stats["saved_policy_trajectories"] += int(not trajectory_had_intvn)
        actor_stats["saved_transitions"] += len(trajectory)
        actor_stats["saved_intervention_samples"] += len(intervention_samples)
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
            "buffer/actor_saved_policy_trajectories": actor_stats[
                "saved_policy_trajectories"
            ],
            "buffer/actor_saved_transitions": actor_stats["saved_transitions"],
            "buffer/actor_saved_intervention_samples": actor_stats[
                "saved_intervention_samples"
            ],
            "buffer/last_trajectory_length": len(trajectory),
            "buffer/last_trajectory_success": int(success),
            "buffer/last_trajectory_intervention": int(trajectory_had_intvn),
            "buffer/last_trajectory_intervention_samples": len(intervention_samples),
        }
        try:
            client.request("send-stats", stats_payload)
        except Exception as exc:
            print(f"actor stats log failed: {exc}", flush=True)
        trajectory_index += 1
        print(
            f"saved trajectory step={step} len={len(trajectory)} label={label} "
            f"intervention={trajectory_had_intvn}",
            flush=True,
        )
        return len(trajectory)

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

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    try:
        for step in pbar:
            last_seen_step = step
            timer.tick("total")
            apply_pending_network()

            manual_label = manual_labeler.consume()
            if manual_label is not None:
                if not current_trajectory:
                    print(
                        f"ignored stale manual {manual_label} label after reset",
                        flush=True,
                    )
                    timer.tock("total")
                    continue
                prepare_manual_reset(env, action_filter)
                baseline_recovery.reset()
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
                data_store.insert(transition)
                current_trajectory.append(copy.deepcopy(transition))
                dump_trajectory(current_trajectory, step)
                current_trajectory = []
                sync_actor_data(step, force=True)
                running_return += reward
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                print(
                    f"manual {manual_label} marked; resetting environment", flush=True
                )
                print("[actor] calling env.reset", flush=True)
                obs, _ = env.reset()
                manual_labeler.clear()
                print("[actor] env.reset returned", flush=True)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                trajectory_had_intervention = False
                action_filter.reset()
                baseline_recovery.reset()
                timer.tock("total")
                continue

            if manual_labeler.consume_baseline_toggle():
                baseline_recovery.toggle_runtime()

            with timer.context("sample_actions"):
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
                spacemouse_action = _peek_spacemouse_intervention(env)
                if spacemouse_action is not None:
                    actions = clip_action_to_space(spacemouse_action, env.action_space)
                    baseline_recovery.reset()
                    baseline_meta = {
                        "baseline_recovery_skipped_for_spacemouse": True,
                        "action_source": "spacemouse",
                    }
                else:
                    actions, baseline_meta = baseline_recovery.select_action(
                        obs, actions, actor_step=step, episode_id=trajectory_index
                    )
                    action_source = baseline_meta.get("action_source", "policy")
                    if action_source == "baseline_recovery" or (
                        action_source == "policy" and step >= config.random_steps
                    ):
                        actions = action_filter(actions)

            # Step environment
            with timer.context("step_env"):
                next_obs, reward, done, truncated, info = env.step(actions)
                info.update(baseline_meta)
                if "left" in info:
                    info.pop("left")
                if "right" in info:
                    info.pop("right")

                if "intervene_action" in info:
                    actions = info.pop("intervene_action")
                    actions = clip_action_to_space(actions, env.action_space)
                    baseline_recovery.reset()
                    info["baseline_recovery_overridden_by_spacemouse"] = True
                    print(
                        "[BASELINE RECOVERY OVERRIDDEN] SpaceMouse intervention detected; "
                        "RecoveryNet state reset.",
                        flush=True,
                    )
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

                manual_label = manual_labeler.consume()
                if manual_label == "success":
                    prepare_manual_reset(env, action_filter)
                    baseline_recovery.reset()
                    reward = 1.0
                    done = True
                    truncated = False
                    info["succeed"] = True
                    info.setdefault("episode", {})
                    print("manual success marked; resetting environment")
                elif manual_label == "failure":
                    prepare_manual_reset(env, action_filter)
                    baseline_recovery.reset()
                    reward = 0.0
                    done = True
                    truncated = False
                    info["succeed"] = False
                    info.setdefault("episode", {})
                    print("manual failure marked; resetting environment")

                terminal = bool(done or truncated)
                if terminal:
                    info.setdefault("succeed", bool(reward > 0.0 and not truncated))
                    info.setdefault("episode", {})
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                running_return += reward
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
                data_store.insert(transition)
                current_trajectory.append(copy.deepcopy(transition))
                if current_intervention:
                    intvn_data_store.insert(copy.deepcopy(transition))

                obs = next_obs
                sync_actor_data(step)

                if done or truncated:
                    info.setdefault("episode", {})
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                    pbar.set_description(f"last return: {running_return}")
                    dump_trajectory(current_trajectory, step)
                    current_trajectory = []
                    sync_actor_data(step, force=True)
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
                    trajectory_had_intervention = False
                    prepare_manual_reset(env, action_filter)
                    baseline_recovery.reset()
                    print("[actor] calling env.reset", flush=True)
                    obs, _ = env.reset()
                    manual_labeler.clear()
                    print("[actor] env.reset returned", flush=True)
                    action_filter.reset()

            if (
                config.buffer_period > 0
                and step - last_buffer_dump_step >= config.buffer_period
            ):
                dump_actor_buffers(step)

            timer.tock("total")

            # Keep the real-time actor loop free of blocking trainer requests.
            # Episode stats are sent after reset, where a short trainer timeout is less harmful.
    finally:
        try:
            pbar.close()
        except Exception:
            pass
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


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
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
    server.start(threaded=True)

    # Publish immediately so actor can start later without blocking learner startup.
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

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
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)
            wandb_logger.log(
                {
                    "buffer/online_size": len(replay_buffer),
                    "buffer/demo_size": len(demo_buffer),
                    "buffer/online_ready": int(
                        len(replay_buffer) >= config.training_starts
                    ),
                    "buffer/using_online_replay": int(using_online_replay),
                    "buffer/training_starts": config.training_starts,
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


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

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
        if latest_ckpt or buffer_files or demo_buffer_files:
            raise FileExistsError(
                f"Checkpoint path already has training state: {FLAGS.checkpoint_path}. "
                "Use --resume_training to continue it, or use a new --checkpoint_path."
            )

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
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
            project="hil-serl",
            description=wandb_description,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

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

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    print(os.getcwd())
    app.run(main)
