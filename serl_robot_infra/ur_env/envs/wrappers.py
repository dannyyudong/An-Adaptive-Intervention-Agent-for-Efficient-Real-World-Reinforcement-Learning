import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from ur_env.spacemouse.spacemouse_expert import SpaceMouseExpert
import time
from scipy.spatial.transform import Rotation as R
from typing import Tuple
from ur_env.utils.rotations import quat_2_euler, quat_2_mrp, quat_2_rotvec

from ur_env.spacemouse.fake_spacemouse import FakeSpaceMouseExpert

ROT90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
ROT_GENERAL = np.array([np.eye(3), ROT90, ROT90 @ ROT90, ROT90.transpose()])


def quat_diff(quat1: np.ndarray, quat2: np.ndarray) -> np.ndarray:
    quat1 = R.from_quat(quat1)
    quat2 = R.from_quat(quat2)
    rel = quat2 * quat1.inv()
    return rel.as_quat()


class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, gripper_action_span=3, device_number: int = 0):
        super().__init__(env)
        self.action_dim = int(np.prod(self.env.action_space.shape))
        cfg = getattr(self.unwrapped, "config", None)
        self.gripper_enabled = self.action_dim > 6 and bool(
            getattr(cfg, "SPACEMOUSE_GRIPPER_ENABLED", True)
        )
        # Mask is applied to the final action for both policy and SpaceMouse paths.
        self.executed_action_mask = np.ones(self.action_dim, dtype=np.float32)
        try:
            if cfg is not None:
                mask = getattr(
                    cfg,
                    "EXECUTED_ACTION_MASK",
                    getattr(cfg, "POLICY_ACTION_MASK", self.executed_action_mask),
                )
                self.executed_action_mask = np.asarray(mask, dtype=np.float32).reshape(
                    -1
                )
        except Exception:
            pass

        try:
            self.expert = SpaceMouseExpert()
        except Exception as e:
            self.expert = FakeSpaceMouseExpert()
            print(f"openend fake SpacemouseExpert since: {e}")

        self.last_intervene = 0
        self._pending_intervention_action = None
        self._allow_full_action_once = False
        self.left = np.array([False] * gripper_action_span, dtype=np.bool_)
        self.right = self.left.copy()
        self.left_button_index = (
            int(getattr(cfg, "SPACEMOUSE_LEFT_BUTTON_INDEX", 0))
            if cfg is not None
            else 0
        )
        self.right_button_index = (
            int(getattr(cfg, "SPACEMOUSE_RIGHT_BUTTON_INDEX", -1))
            if cfg is not None
            else -1
        )

        self.invert_axes = np.array([-1, 1, -1, -1, 1, -1], dtype=np.float32)
        if cfg is not None:
            self.invert_axes = np.asarray(
                getattr(cfg, "SPACEMOUSE_INVERT_AXES", self.invert_axes),
                dtype=np.float32,
            ).reshape(6)

        self.deadspace = 0.08

    @staticmethod
    def _button_pressed(buttons, index: int) -> bool:
        try:
            values = np.asarray(buttons, dtype=np.bool_).reshape(-1)
        except Exception:
            values = np.asarray(list(buttons or []), dtype=np.bool_).reshape(-1)
        if index < 0:
            index += int(values.shape[0])
        if index < 0 or index >= values.shape[0]:
            return False
        return bool(values[index])

    def clear_intervention(self):
        self.last_intervene = 0
        self._pending_intervention_action = None
        self._allow_full_action_once = False
        self.left[:] = False
        self.right[:] = False

    def allow_full_action_once(self):
        """Bypass the configured mask for the next non-human action only."""
        self._allow_full_action_once = True

    def action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a = self.get_deadspace_action()

        if (
            np.linalg.norm(expert_a) > 0.001 or self.left.any() or self.right.any()
        ):  # also read buttons with no movement
            self.last_intervene = time.time()

        if self.gripper_enabled:
            gripper_action = (
                np.zeros((1,)) + int(self.left.any()) - int(self.right.any())
            )
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if time.time() - self.last_intervene < 0.5:
            expert_a = self.adapt_spacemouse_output(expert_a)
            expert_a = self._match_action_dim(expert_a)
            return expert_a, True

        return action, False

    def get_deadspace_action(self) -> np.ndarray:
        expert_a, buttons = self.expert.get_action()

        positive = np.clip(
            (expert_a - self.deadspace) / (1.0 - self.deadspace), a_min=0.0, a_max=1.0
        )
        negative = np.clip(
            (expert_a + self.deadspace) / (1.0 - self.deadspace), a_min=-1.0, a_max=0.0
        )
        expert_a = positive + negative

        self.left, self.right = np.roll(self.left, -1), np.roll(
            self.right, -1
        )  # shift them one to the left
        self.left[-1] = self._button_pressed(buttons, self.left_button_index)
        self.right[-1] = self._button_pressed(buttons, self.right_button_index)

        return np.array(expert_a, dtype=np.float32)

    def adapt_spacemouse_output(self, action: np.ndarray) -> np.ndarray:
        position = self.unwrapped.curr_pos
        # print(position)
        # Extract the actual TCP orientation as a quaternion (the last 4 elements)
        tcp_quat = position[3:]
        tcp_rot = R.from_quat(tcp_quat)

        action[:6] *= self.invert_axes

        # Apply the actual gripper orientation to the spacemouse actions
        action[:3] = tcp_rot.apply(action[:3])  # Translation local to the gripper tip
        action[3:6] = tcp_rot.apply(action[3:6])  # Rotation local to the gripper tip

        return action

    def _match_action_dim(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] == self.action_dim:
            return action
        if action.shape[0] > self.action_dim:
            return action[: self.action_dim]

        full_action = np.zeros((self.action_dim,), dtype=np.float32)
        full_action[: action.shape[0]] = action
        return full_action

    def _apply_executed_action_mask(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        mask = self.executed_action_mask
        if mask.shape[0] < action.shape[0]:
            mask = np.pad(
                mask, (0, action.shape[0] - mask.shape[0]), constant_values=1.0
            )
        return action * mask[: action.shape[0]]

    def peek_intervention_action(self):
        if self._pending_intervention_action is not None:
            return self._pending_intervention_action.copy()

        zero_action = np.zeros(self.action_space.shape, dtype=np.float32)
        new_action, replaced = self.action(zero_action)
        if not replaced:
            return None

        new_action = self._match_action_dim(new_action)
        new_action = self._apply_executed_action_mask(new_action)
        self._pending_intervention_action = np.asarray(
            new_action, dtype=np.float32
        ).copy()
        return self._pending_intervention_action.copy()

    def step(self, action):
        allow_full_action = self._allow_full_action_once
        self._allow_full_action_once = False
        if self._pending_intervention_action is not None:
            new_action = self._match_action_dim(self._pending_intervention_action)
            self._pending_intervention_action = None
            replaced = True
        else:
            new_action, replaced = self.action(action)
            new_action = self._match_action_dim(new_action)
            if replaced or not allow_full_action:
                new_action = self._apply_executed_action_mask(new_action)
        obs, rew, done, truncated, info = self.env.step(new_action)

        if replaced:
            info["intervene_action"] = new_action
        info["executed_action"] = new_action
        info["left"] = self.left.any()
        info["right"] = self.right.any()
        return obs, rew, done, truncated, info


class Quat2EulerWrapper(
    gym.ObservationWrapper
):  # not used anymore (stay away from euler angles!)
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], quat_2_euler(tcp_pose[3:]))
        )
        return observation


class Quat2MrpWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], quat_2_mrp(tcp_pose[3:]))
        )
        return observation


class Quat2rotvecWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotvec
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], quat_2_rotvec(tcp_pose[3:]))
        )
        return observation


def rotate_state(state: np.ndarray, num_rot: int):
    assert len(state.shape) == 1 and state.shape[0] % 3 == 0
    state = state.reshape((-1, 3)).transpose()
    rotated = np.dot(ROT_GENERAL[num_rot % 4], state).transpose()
    return rotated.reshape((-1))


class ObservationRotationWrapper(gym.Wrapper):
    """
    Convert every observation into the first quadrant of the Relative Frame
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        print("Observation Rotation Wrapper enabled!")
        self.num_rot_quadrant = -1

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        obs = self.rotate_observation(obs, random=True)  # rotate initial state random
        return obs, info

    def step(self, action: np.ndarray):
        action = self.rotate_action(action=action)
        obs, reward, done, truncated, info = self.env.step(action)
        # print("\nquadrant: ", self.num_rot_quadrant)
        rotated_obs = self.rotate_observation(obs)
        return rotated_obs, reward, done, truncated, info

    def rotate_observation(self, observation, random=False):
        if not random:
            x, y = observation["state"]["tcp_pose"][:2]
            self.num_rot_quadrant = int(x < 0.0) * 2 + int(
                x * y < 0.0
            )  # save quadrant info
        else:
            self.num_rot_quadrant = (
                int(time.time_ns()) % 4
            )  # do not mess with seeded np.random

        for state in observation["state"].keys():
            if state == "gripper_state":
                continue
            elif state == "action":
                observation["state"][state][:6] = rotate_state(
                    observation["state"][state][:6], self.num_rot_quadrant
                )
            else:
                observation["state"][state][:] = rotate_state(
                    observation["state"][state], self.num_rot_quadrant
                )  # rotate

        if "images" in observation:
            for image_keys in observation["images"].keys():
                observation["images"][image_keys][:] = np.rot90(
                    observation["images"][image_keys],
                    axes=(0, 1),
                    k=self.num_rot_quadrant,
                )
        return observation

    def rotate_action(self, action):
        rotated_action = action.copy()
        rotated_action[:6] = rotate_state(
            action[:6], 4 - self.num_rot_quadrant
        )  # rotate
        return rotated_action


class GripperCloseEnv(gym.ActionWrapper):
    """
    Policy outputs 6D actions (xyz + rot/mrp), wrapper expands to 7D by
    appending a fixed gripper action.

    By default fixed_gripper_action=0.0 -> "no gripper command".
    """

    def __init__(self, env: gym.Env, fixed_gripper_action: float = 0.0):
        super().__init__(env)

        assert isinstance(
            self.env.action_space, Box
        ), "Underlying action_space must be Box"
        assert self.env.action_space.shape == (
            7,
        ), f"Expected underlying action shape (7,), got {self.env.action_space.shape}"

        self.fixed_gripper_action = float(fixed_gripper_action)

        low = np.asarray(self.env.action_space.low[:6], dtype=np.float32)
        high = np.asarray(self.env.action_space.high[:6], dtype=np.float32)

        self.action_space = Box(low=low, high=high, dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] == 7:
            a = a[:6]
        if a.shape[0] != 6:
            raise ValueError(f"GripperCloseEnv expected 6D action, got shape {a.shape}")

        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = a.copy()

        # clamp fixed gripper value into env bounds just in case
        g_low = float(np.asarray(self.env.action_space.low[6]))
        g_high = float(np.asarray(self.env.action_space.high[6]))
        new_action[6] = float(np.clip(self.fixed_gripper_action, g_low, g_high))
        return new_action

    def step(self, action):
        new_action = self.action(action)

        out = self.env.step(new_action)
        if len(out) == 5:
            obs, rew, terminated, truncated, info = out
        elif len(out) == 4:
            # fallback for older gym envs
            obs, rew, done, info = out
            terminated, truncated = bool(done), False
        else:
            raise RuntimeError(f"Unexpected env.step() return length: {len(out)}")

        # Keep action metadata consistent with the *policy* action space (6D).
        if isinstance(info, dict):
            for key in ("intervene_action", "executed_action"):
                if key in info and info[key] is not None:
                    ia = np.asarray(info[key])
                    if ia.shape[-1] == 7:
                        info[key] = ia[..., :6]

        if isinstance(info, dict):
            info["fixed_gripper_action"] = self.fixed_gripper_action

        return obs, rew, terminated, truncated, info
