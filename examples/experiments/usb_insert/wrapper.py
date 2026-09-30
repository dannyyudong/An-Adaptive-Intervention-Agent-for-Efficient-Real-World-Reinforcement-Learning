import copy
import time

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from ur_env.envs.ur5_env import UR5Env


class UR7ERAMEnv(UR5Env):
    """UR7e RAM pickup/insertion env using wrist + external RealSense cameras."""

    FAR_X_Z_GUARD_X_MAX = -0.35
    FAR_X_Z_MIN = 0.20
    FAR_X_CLOSE_GRIPPER_ACTION = 1.0
    Z_MAX = 0.40

    def __init__(
        self,
        *,
        config,
        hz=10,
        fake_env=False,
        max_episode_length=150,
        save_video=False,
        camera_mode="rgb",
    ):
        super().__init__(
            hz=hz,
            fake_env=fake_env,
            config=config,
            max_episode_length=max_episode_length,
            save_video=save_video,
            camera_mode=camera_mode,
        )
        self._far_x_close_gripper_requested = False

        if isinstance(self.observation_space, gym.spaces.Dict):
            state_space = self.observation_space.spaces.get("state")
            if isinstance(state_space, gym.spaces.Dict):
                state_space.spaces["gripper_object"] = gym.spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                )

    def _constrain_to_tool_roll(self, pose):
        if not bool(getattr(self.config, "TOOL_ROLL_ONLY", False)):
            return pose

        reset_quat = np.asarray(self.curr_reset_pose[3:], dtype=np.float64)
        if np.linalg.norm(reset_quat) < 1e-6:
            return pose

        reset_rot = R.from_quat(reset_quat)
        candidate_rot = R.from_quat(pose[3:])

        axis_local = np.asarray(
            getattr(self.config, "TOOL_ROLL_AXIS", [0.0, 0.0, 1.0]), dtype=np.float64
        )
        axis_norm = np.linalg.norm(axis_local)
        if axis_norm < 1e-6:
            return pose
        axis_local = axis_local / axis_norm
        axis_base = reset_rot.apply(axis_local)
        axis_base = axis_base / np.linalg.norm(axis_base)

        rel_rot = candidate_rot * reset_rot.inv()
        roll_angle = float(np.dot(rel_rot.as_rotvec(), axis_base))
        constrained_rot = R.from_rotvec(axis_base * roll_angle) * reset_rot
        pose[3:] = constrained_rot.as_quat()
        return pose

    def _z_max(self) -> float:
        return float(getattr(self.config, "Z_MAX", self.Z_MAX))

    def clip_safety_box(self, next_pos: np.ndarray) -> np.ndarray:
        next_pos = super().clip_safety_box(next_pos)
        far_x = float(next_pos[0]) < self.FAR_X_Z_GUARD_X_MAX
        at_z_floor = float(next_pos[2]) <= self.FAR_X_Z_MIN
        self._far_x_close_gripper_requested = bool(far_x and at_z_floor)
        if far_x and float(next_pos[2]) < self.FAR_X_Z_MIN:
            next_pos[2] = self.FAR_X_Z_MIN
        next_pos[2] = min(float(next_pos[2]), self._z_max())
        return self._constrain_to_tool_roll(next_pos)

    def _send_gripper_command(self, gripper_pos: np.ndarray):
        if bool(getattr(self, "_far_x_close_gripper_requested", False)):
            gripper_pos = np.asarray(
                [self.FAR_X_CLOSE_GRIPPER_ACTION], dtype=np.float32
            )
            self._far_x_close_gripper_requested = False
        return super()._send_gripper_command(gripper_pos)

    def hold_position(self, stop_rtde=False):
        """Best-effort stop: hold the current TCP pose before reset or shutdown."""
        try:
            self._update_currpos()
            self.controller.set_target_pos(self.curr_pos.copy())
            if hasattr(self.controller, "set_gripper_pos"):
                self.controller.set_gripper_pos(np.zeros((1,), dtype=np.float32))
        except Exception as exc:
            print(f"hold_position target update failed: {exc}")

        if not stop_rtde:
            return

        ur_control = getattr(self.controller, "ur_control", None)
        if ur_control is not None:
            try:
                ur_control.forceModeStop()
            except Exception:
                pass
            try:
                ur_control.servoStop()
            except Exception:
                pass
            try:
                ur_control.speedStop(a=1.5)
            except Exception:
                pass

    def _pose6_to_pose7(self, pose6):
        pose6 = np.asarray(pose6, dtype=np.float64).reshape(6)
        pose7 = np.empty((7,), dtype=np.float32)
        pose7[:3] = pose6[:3]
        pose7[3:] = R.from_rotvec(pose6[3:]).as_quat()
        return pose7

    def _sample_reset_pose6(self):
        reset_pose = (
            np.asarray(self.config.HOME_TCP_POSE, dtype=np.float64).reshape(-1).copy()
        )
        if reset_pose.shape != (6,):
            raise ValueError(
                f"HOME_TCP_POSE must be xyz+rotvec with shape (6,), got {reset_pose.shape}"
            )

        if bool(getattr(self.config, "RESET_NOISE_ENABLED", False)):
            xyz_noise = np.asarray(
                getattr(self.config, "RESET_XYZ_NOISE", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            ).reshape(3)
            noise = np.random.uniform(-xyz_noise, xyz_noise)
            reset_pose[:3] += noise
            print(
                f"[reset] xyz noise={np.round(noise, 4)} target_xyz={np.round(reset_pose[:3], 4)}",
                flush=True,
            )

        reset_z_max = float(getattr(self.config, "RESET_TARGET_Z_MAX", self._z_max()))
        if float(reset_pose[2]) > reset_z_max:
            print(
                f"[reset] clipping home z from {reset_pose[2]:.4f} to RESET_TARGET_Z_MAX={reset_z_max:.4f}",
                flush=True,
            )
            reset_pose[2] = reset_z_max
        reset_pose[2] = min(float(reset_pose[2]), self._z_max())

        return reset_pose

    def _validate_reset_target(self, target_pose, label):
        max_translation = float(getattr(self.config, "RESET_MAX_TRANSLATION", 0.08))
        max_rot_delta = float(getattr(self.config, "RESET_MAX_ROT_DELTA", 0.35))
        pos_delta = float(np.linalg.norm(target_pose[:3] - self.curr_pos[:3]))
        rot_delta = float(
            (
                R.from_quat(target_pose[3:]) * R.from_quat(self.curr_pos[3:]).inv()
            ).magnitude()
        )
        if pos_delta > max_translation or rot_delta > max_rot_delta:
            print(
                f"[reset] refusing unsafe {label} target: pos_delta={pos_delta:.4f} "
                f"limit={max_translation:.4f}, rot_delta={rot_delta:.4f} limit={max_rot_delta:.4f}, "
                f"curr={np.round(self.curr_pos, 4)}, target={np.round(target_pose, 4)}",
                flush=True,
            )
            self.hold_position()
            return False
        return True

    def _set_reset_motion_profile(self):
        """Temporarily make force-mode tracking faster during reset only."""
        controller = getattr(self, "controller", None)
        if controller is None:
            return {}

        multiplier = float(getattr(self.config, "RESET_SPEED_MULTIPLIER", 1.0))
        limit_scale = float(getattr(self.config, "RESET_FORCE_LIMIT_SCALE", multiplier))
        multiplier = max(1.0, multiplier)
        limit_scale = max(1.0, limit_scale)

        saved = {}
        for attr in ("kp", "kd", "rot_kp", "rot_kd", "delta"):
            if hasattr(controller, attr):
                value = getattr(controller, attr)
                saved[attr] = value
                setattr(controller, attr, float(value) * multiplier)

        if hasattr(controller, "fm_damping"):
            saved["fm_damping"] = controller.fm_damping
            controller.fm_damping = float(controller.fm_damping) / multiplier

        if hasattr(controller, "fm_limits"):
            saved["fm_limits"] = np.asarray(
                controller.fm_limits, dtype=np.float32
            ).copy()
            controller.fm_limits = saved["fm_limits"] * limit_scale

        return saved

    def _restore_motion_profile(self, saved):
        controller = getattr(self, "controller", None)
        if controller is None:
            return
        for attr, value in saved.items():
            if attr == "fm_limits":
                controller.fm_limits = value.copy()
            else:
                setattr(controller, attr, value)

    def _apply_reset_gripper_action(self):
        action_name = str(getattr(self.config, "RESET_GRIPPER_ACTION", "none")).lower()
        action_map = {
            "none": 0.0,
            "hold": 0.0,
            "release": -1.0,
            "open": -1.0,
            "close": 1.0,
            "grip": 1.0,
        }
        if action_name not in action_map:
            raise ValueError(f"Unknown RESET_GRIPPER_ACTION: {action_name}")
        action = action_map[action_name]
        if abs(action) < 1e-6:
            return
        try:
            self._send_gripper_command(np.asarray([action], dtype=np.float32))
            print(f"[reset] gripper action={action_name}", flush=True)
            time.sleep(float(getattr(self.config, "RESET_GRIPPER_SETTLE_S", 0.15)))
        except Exception as exc:
            print(f"[reset] gripper action failed: {exc}", flush=True)

    def _servo_to_pose(self, target_pose, label):
        target_pose = np.asarray(target_pose, dtype=np.float32).reshape(7)
        saved_profile = self._set_reset_motion_profile()
        try:
            self._update_currpos()
            if not self._validate_reset_target(target_pose, label):
                return False
            self.controller.set_target_pos(target_pose)
            timeout_s = float(getattr(self.config, "RESET_TIMEOUT_S", 4.0))
            pos_tol = float(getattr(self.config, "RESET_POS_TOL", 0.0015))
            rot_tol = float(getattr(self.config, "RESET_ROT_TOL", 0.03))
            wait_dt = float(getattr(self.config, "RESET_WAIT_DT", 0.005))
            deadline = time.monotonic() + timeout_s
            last_report_at = 0.0
            print(
                f"[reset] servo {label} target={np.round(target_pose, 4)} "
                f"timeout={timeout_s:.1f}s pos_tol={pos_tol:.4f} rot_tol={rot_tol:.4f}",
                flush=True,
            )

            while True:
                self._update_currpos()
                pos_err = float(np.linalg.norm(target_pose[:3] - self.curr_pos[:3]))
                rot_err = float(
                    (
                        R.from_quat(target_pose[3:])
                        * R.from_quat(self.curr_pos[3:]).inv()
                    ).magnitude()
                )
                if pos_err <= pos_tol and rot_err <= rot_tol:
                    print(
                        f"[reset] reached {label}: pos_err={pos_err:.4f}, rot_err={rot_err:.4f}, "
                        f"curr={np.round(self.curr_pos, 4)}",
                        flush=True,
                    )
                    return True
                now = time.monotonic()
                if now - last_report_at >= 0.5:
                    last_report_at = now
                    print(
                        f"[reset] waiting for {label}: pos_err={pos_err:.4f}, rot_err={rot_err:.4f}, "
                        f"curr={np.round(self.curr_pos, 4)}, target={np.round(target_pose, 4)}",
                        flush=True,
                    )
                if now > deadline:
                    print(
                        f"[reset] timeout waiting for {label} after {timeout_s:.1f}s; "
                        f"pos_err={pos_err:.4f}, rot_err={rot_err:.4f}, "
                        f"curr={np.round(self.curr_pos, 4)}, target={np.round(target_pose, 4)}",
                        flush=True,
                    )
                    self.hold_position()
                    return False
                time.sleep(wait_dt)
        finally:
            self._restore_motion_profile(saved_profile)

    def _servo_path_to_pose(self, target_pose, label):
        """Servo to a distant reset target through small validated segments."""
        target_pose = np.asarray(target_pose, dtype=np.float32).reshape(7)
        self._update_currpos()
        start_pose = self.curr_pos.copy()

        max_translation = max(
            float(getattr(self.config, "RESET_MAX_TRANSLATION", 0.08)) * 0.75, 0.01
        )
        max_rot_delta = max(
            float(getattr(self.config, "RESET_MAX_ROT_DELTA", 0.35)) * 0.75, 0.05
        )
        pos_delta = float(np.linalg.norm(target_pose[:3] - start_pose[:3]))
        rot_delta = float(
            (
                R.from_quat(target_pose[3:]) * R.from_quat(start_pose[3:]).inv()
            ).magnitude()
        )
        steps = int(
            max(
                1,
                np.ceil(pos_delta / max_translation),
                np.ceil(rot_delta / max_rot_delta),
            )
        )

        if steps > 1:
            print(
                f"[reset] {label} split into {steps} segments: "
                f"pos_delta={pos_delta:.4f}, rot_delta={rot_delta:.4f}",
                flush=True,
            )

        slerp = Slerp([0.0, 1.0], R.from_quat([start_pose[3:], target_pose[3:]]))
        for idx in range(1, steps + 1):
            frac = idx / steps
            waypoint = np.empty((7,), dtype=np.float32)
            waypoint[:3] = start_pose[:3] + (target_pose[:3] - start_pose[:3]) * frac
            waypoint[3:] = slerp([frac]).as_quat()[0]
            waypoint[2] = min(float(waypoint[2]), self._z_max())
            if not self._servo_to_pose(waypoint, f"{label} {idx}/{steps}"):
                print(
                    f"[reset] failed during {label} segment {idx}/{steps}", flush=True
                )
                return False
        return True

    def close(self):
        """Stop the controller without leaving a stale force-mode command active."""
        try:
            self.hold_position(stop_rtde=True)
        except Exception as exc:
            print(f"close hold_position failed: {exc}")

        controller = getattr(self, "controller", None)
        if controller is not None:
            try:
                controller.stop()
            except Exception as exc:
                print(f"controller stop failed: {exc}")
            try:
                if hasattr(controller, "join"):
                    controller.join(timeout=2.0)
            except Exception as exc:
                print(f"controller join failed: {exc}")

        try:
            self.close_cameras()
        except Exception as exc:
            print(f"camera cleanup failed: {exc}")

    def go_to_rest(self):
        """Reset by servoing the force-mode target: lift in +z, then return home."""
        reset_pose = self._sample_reset_pose6()

        print("[reset] lift +z then servo home", flush=True)
        self.hold_position()
        print("[reset] hold current pose", flush=True)
        self._apply_reset_gripper_action()
        self._update_currpos()

        lift_z = float(getattr(self.config, "RESET_LIFT_Z", 0.05))
        if lift_z > 0.0:
            lift_pose = self.curr_pos.copy()
            lift_pose[2] = min(float(lift_pose[2] + lift_z), self._z_max())
            if not self._servo_to_pose(lift_pose, "lift"):
                self.curr_reset_pose[:] = self.curr_pos
                return np.zeros((2,))

        home_pose = self._pose6_to_pose7(reset_pose)
        home_pose[2] = min(float(home_pose[2]), self._z_max())
        if not self._servo_path_to_pose(home_pose, "home"):
            self.curr_reset_pose[:] = self.curr_pos
            print("[reset] home servo failed; holding current pose", flush=True)
            return np.zeros((2,))

        self.curr_reset_pose[:] = home_pose
        print("[reset] done", flush=True)
        return np.zeros((2,))

    def _get_obs(self, action) -> dict:
        obs = super()._get_obs(action)
        obs["state"]["gripper_object"] = np.asarray(
            [self.gripper_state[1]], dtype=np.float32
        )
        return obs


class XYZGraspActionWrapper(gym.ActionWrapper):
    """Expose [x, y, z, gripper] while executing zero rotation commands."""

    FULL_ACTION_DIM = 7
    ACTION_INDICES = np.array([0, 1, 2, 6], dtype=np.int64)

    def __init__(self, env):
        super().__init__(env)
        low = np.asarray(env.action_space.low, dtype=np.float32)
        high = np.asarray(env.action_space.high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=low[self.ACTION_INDICES],
            high=high[self.ACTION_INDICES],
            dtype=np.float32,
        )

    def action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        full_action = np.zeros(self.FULL_ACTION_DIM, dtype=np.float32)
        full_action[self.ACTION_INDICES] = action[: self.ACTION_INDICES.shape[0]]
        return full_action

    def _compress_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] >= self.FULL_ACTION_DIM:
            return action[self.ACTION_INDICES].astype(np.float32)
        return action.astype(np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self.action(action))
        info = copy.deepcopy(info)
        for key in ("intervene_action", "executed_action"):
            if key in info:
                info[key] = self._compress_action(info[key])
        return obs, reward, terminated, truncated, info


class GripperPenaltyWrapper(gym.Wrapper):
    """Adds grasp_penalty expected by the hybrid single-arm SAC agent."""

    def __init__(self, env, penalty=-0.02):
        super().__init__(env)
        self.penalty = float(penalty)
        self.last_closed_norm = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_closed_norm = float(self.env.unwrapped.gripper_state[0])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        effective_action = info.get("intervene_action", action)
        effective_action = np.asarray(effective_action, dtype=np.float32).reshape(-1)

        closed_norm = float(self.env.unwrapped.gripper_state[0])
        toggling = False
        if self.last_closed_norm is not None and effective_action.shape[0] > 3:
            toggling = (
                effective_action[-1] < -0.5 and self.last_closed_norm < 0.1
            ) or (effective_action[-1] > 0.5 and self.last_closed_norm > 0.9)

        info = copy.deepcopy(info)
        info["grasp_penalty"] = self.penalty if toggling else 0.0
        self.last_closed_norm = closed_norm
        return obs, reward, terminated, truncated, info
