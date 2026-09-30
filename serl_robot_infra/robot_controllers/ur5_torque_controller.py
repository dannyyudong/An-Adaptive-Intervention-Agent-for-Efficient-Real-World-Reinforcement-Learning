import time
import threading
import asyncio
import numpy as np
from scipy.spatial.transform import Rotation as R

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from ur_env.utils.rotations import pose2quat, rotvec_2_quat
from ur_env.utils.robotiq_usb import Robotiq2F85USBGripper as Robotiq2F85Gripper


class UrDirectTorqueController(threading.Thread):
    """
    Drop-in replacement for your UrImpedanceController that:
      - accepts target TCP pose via set_target_pos() (same as your env uses)
      - computes joint torques using task-space impedance: tau = J^T * wrench - Kq*qd
      - sends torques via RTDE direct torque API if available

    Compatibility:
      - UR5Env can keep calling controller.set_target_pos(next_pos)
      - Spacemouse wrapper unchanged (still outputs 7D action to env)
    """

    def __init__(
        self,
        robot_ip: str,
        frequency: int = 250,  # torque control wants higher freq; use 250 if stable; else 125
        config=None,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if config is None:
            raise ValueError("config is required")

        # Thread flags
        self._stop = threading.Event()
        self._reset = threading.Event()
        self._is_ready = threading.Event()
        self._is_truncated = threading.Event()
        self.lock = threading.Lock()

        self.robot_ip = robot_ip
        self.frequency = int(frequency)
        self.dt = 1.0 / float(self.frequency)
        self.verbose = bool(verbose)
        self.config = config

        # --- Target state (same semantics as your current controller) ---
        self.target_pos = np.zeros((7,), dtype=np.float32)  # [x,y,z,qx,qy,qz,qw]
        self.target_grip = np.zeros((1,), dtype=np.float32)

        # --- Measured state ---
        self.curr_pos = np.zeros((7,), dtype=np.float32)
        self.curr_vel = np.zeros((6,), dtype=np.float32)  # TCP twist
        self.curr_Q = np.zeros((6,), dtype=np.float32)
        self.curr_Qd = np.zeros((6,), dtype=np.float32)

        self.curr_force = np.zeros((6,), dtype=np.float32)  # if available
        self.gripper_state = np.zeros((2,), dtype=np.float32)

        # --- Reset config ---
        if hasattr(config, "RESET_Q"):
            self.reset_Q = np.asarray(config.RESET_Q).reshape(-1).astype(np.float32)
            if self.reset_Q.shape[0] != 6:
                raise ValueError(
                    f"RESET_Q must have 6 joints, got {self.reset_Q.shape}"
                )
        else:
            self.reset_Q = np.deg2rad(
                np.array([272.0, -91.0, -130.0, -166.0, 270.0, 0.0], dtype=np.float32)
            )

        self.reset_pose = np.zeros((6,), dtype=np.float32)  # optional pose reset
        self.reset_height = float(getattr(config, "RESET_HEIGHT", 0.10))

        # --- Safety / stability knobs (VERY IMPORTANT for torque control) ---
        # Pose error clipping (prevents huge torques if target jumps)
        self.max_ep = float(getattr(config, "TORQUE_MAX_POS_ERR", 0.02))  # m
        self.max_er = float(
            getattr(config, "TORQUE_MAX_ROT_ERR", 0.25)
        )  # rad (rotvec magnitude)

        # Task-space impedance gains (tune carefully!)
        self.Kp_pos = float(getattr(config, "TORQUE_KP_POS", 250.0))  # N/m-ish scale
        self.Kd_pos = float(getattr(config, "TORQUE_KD_POS", 30.0))  # N/(m/s)
        self.Kp_rot = float(getattr(config, "TORQUE_KP_ROT", 25.0))  # Nm/rad
        self.Kd_rot = float(getattr(config, "TORQUE_KD_ROT", 2.0))  # Nm/(rad/s)

        # Joint damping
        self.Kq = float(getattr(config, "TORQUE_KQ", 0.6))

        # Torque limits (start conservative)
        # If config provides per-joint limits, use them; else default small-ish.
        self.tau_limit = np.asarray(
            getattr(
                config, "TORQUE_LIMITS", np.array([8, 8, 8, 4, 4, 3], dtype=np.float32)
            ),
            dtype=np.float32,
        ).reshape(
            6,
        )

        # Torque rate limit (Nm/s) to avoid jerks
        self.tau_rate = float(getattr(config, "TORQUE_RATE_LIMIT", 80.0))  # Nm/s
        self._tau_prev = np.zeros((6,), dtype=np.float32)

        # Watchdog: if no target update recently, send 0 torque
        self.watchdog_s = float(getattr(config, "TORQUE_WATCHDOG_S", 0.25))
        self._last_target_time = time.monotonic()

        # RTDE
        self.ur_control: RTDEControlInterface = None
        self.ur_receive: RTDEReceiveInterface = None

        # Gripper
        self.robotiq_gripper: Robotiq2F85Gripper = None
        self.gripper_timeout = {
            "timeout": int(getattr(config, "GRIPPER_TIMEOUT", 5000)),
            "last_grip": time.monotonic() - 1e6,
        }

    # ------------------------- Public API (compatible) -------------------------

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.is_set()

    def is_ready(self):
        return self._is_ready.is_set()

    def is_reset(self):
        return not self._reset.is_set()

    def is_truncated(self):
        return self._is_truncated.is_set()

    def set_target_pos(self, target_pos: np.ndarray):
        """Same interface as your existing controller."""
        if target_pos.shape == (6,):
            # xyz + rotvec -> xyz + quat
            q = rotvec_2_quat(target_pos[3:])
            tp = np.concatenate([target_pos[:3], q], axis=0)
        elif target_pos.shape == (7,):
            tp = target_pos
        else:
            raise ValueError(f"target_pos must be (6,) or (7,), got {target_pos.shape}")

        with self.lock:
            self.target_pos[:] = np.asarray(tp, dtype=np.float32)
            self._last_target_time = time.monotonic()

    def set_gripper_pos(self, target_grip: np.ndarray):
        with self.lock:
            self.target_grip[:] = np.asarray(target_grip, dtype=np.float32)

    def set_reset_Q(self, reset_Q: np.ndarray):
        with self.lock:
            self.reset_Q[:] = np.asarray(reset_Q, dtype=np.float32).reshape(
                6,
            )
        self._reset.set()

    def set_reset_pose(self, reset_pose: np.ndarray):
        with self.lock:
            self.reset_pose[:] = np.asarray(reset_pose, dtype=np.float32).reshape(
                6,
            )
        self._reset.set()

    def get_target_pos(self, copy=True):
        with self.lock:
            return self.target_pos.copy() if copy else self.target_pos

    def get_state(self):
        with self.lock:
            return {
                "pos": self.curr_pos.copy(),
                "vel": self.curr_vel.copy(),
                "Q": self.curr_Q.copy(),
                "Qd": self.curr_Qd.copy(),
                "force": self.curr_force[:3].copy(),
                "torque": self.curr_force[3:].copy(),
                "gripper": self.gripper_state.copy(),
            }

    # ------------------------- RTDE helpers -------------------------

    async def start_ur_interfaces(self, gripper=True):
        self.ur_control = RTDEControlInterface(self.robot_ip)
        self.ur_receive = RTDEReceiveInterface(self.robot_ip)

        # Set TCP offset (keep same as you had)
        self.ur_control.setTcp([0.0, 0.0, 0.16, 0.0, 0.0, 0.0])

        if gripper:
            self.robotiq_gripper = Robotiq2F85Gripper(
                portname=self.config.GRIPPER_USB_PORT,
                slaveaddress=getattr(self.config, "GRIPPER_SLAVE_ID", 9),
                emulate_vacuum_pressure=False,
            )
            await self.robotiq_gripper.connect()
            await self.robotiq_gripper.activate()

        # Stop any lingering modes (important after crashes)
        try:
            self.ur_control.forceModeStop()
        except Exception:
            pass
        try:
            self.ur_control.servoStop()
        except Exception:
            pass
        try:
            self.ur_control.speedStop()
        except Exception:
            pass

    async def _update_robot_state(self):
        # TCP pose & twist
        tcp_pose = self.ur_receive.getActualTCPPose()  # [x,y,z,rx,ry,rz] in axis-angle
        tcp_speed = self.ur_receive.getActualTCPSpeed()  # [vx,vy,vz,wx,wy,wz]
        q = self.ur_receive.getActualQ()
        qd = self.ur_receive.getActualQd()

        # forces (optional)
        try:
            ft = self.ur_receive.getActualTCPForce()  # [Fx,Fy,Fz,Tx,Ty,Tz]
        except Exception:
            ft = [0, 0, 0, 0, 0, 0]

        # gripper state (optional)
        closed_norm = 0.0
        obj_detected = 0.0
        if self.robotiq_gripper is not None:
            grip_pos = await self.robotiq_gripper.get_current_pressure()
            obj = await self.robotiq_gripper.get_object_status()
            closed_norm = float(grip_pos) / 255.0
            obj_detected = 1.0 if obj.value in (1, 2) else 0.0

        with self.lock:
            self.curr_pos[:] = pose2quat(tcp_pose)  # convert to [x,y,z,qx,qy,qz,qw]
            self.curr_vel[:] = np.asarray(tcp_speed, np.float32)
            self.curr_Q[:] = np.asarray(q, np.float32)
            self.curr_Qd[:] = np.asarray(qd, np.float32)
            self.curr_force[:] = np.asarray(ft, np.float32)
            self.gripper_state[:] = np.array(
                [closed_norm, obj_detected], dtype=np.float32
            )

    def _get_jacobian(self) -> np.ndarray | None:
        """
        Robust Jacobian getter: RTDE versions differ.
        Prefer ur_receive.getJacobian() if present.
        """
        if hasattr(self.ur_receive, "getJacobian"):
            J = np.array(self.ur_receive.getJacobian(), dtype=np.float64).reshape(6, 6)
            return J
        if hasattr(self.ur_control, "getJacobian"):
            J = np.array(self.ur_control.getJacobian(), dtype=np.float64).reshape(6, 6)
            return J
        return None

    def _pose_error(self, target_pose7: np.ndarray, curr_pose7: np.ndarray):
        ep = target_pose7[:3] - curr_pose7[:3]
        R_err = R.from_quat(target_pose7[3:]) * R.from_quat(curr_pose7[3:]).inv()
        er = R_err.as_rotvec()
        return ep, er

    def _clip_pose_error(self, ep, er):
        ep = np.clip(ep, -self.max_ep, self.max_ep)
        # clip rotation error by magnitude
        n = np.linalg.norm(er)
        if n > self.max_er:
            er = er * (self.max_er / (n + 1e-9))
        return ep, er

    def _rate_limit_tau(self, tau: np.ndarray) -> np.ndarray:
        # limit per-step change: |Δτ| <= tau_rate * dt
        max_d = self.tau_rate * self.dt
        tau = np.asarray(tau, dtype=np.float32)
        d = np.clip(tau - self._tau_prev, -max_d, max_d)
        out = self._tau_prev + d
        self._tau_prev = out
        return out

    def _send_joint_torques(self, tau6: np.ndarray) -> bool:
        """
        Call your RTDE torque function robustly.
        Names differ: directTorque vs direct_torque vs directTorqueCommand, etc.
        """
        tau6 = np.asarray(tau6, dtype=np.float64).reshape(
            6,
        )
        try:
            if hasattr(self.ur_control, "directTorque"):
                # some builds: directTorque(list)
                self.ur_control.directTorque(tau6.tolist())
                return True
            if hasattr(self.ur_control, "direct_torque"):
                # some wrappers: direct_torque(list)
                self.ur_control.direct_torque(tau6.tolist())
                return True
        except Exception as e:
            if self.verbose:
                print("[TORQUE] send failed:", repr(e))
            return False

        if self.verbose:
            print("[TORQUE] No direct torque method found on RTDEControlInterface.")
        return False

    async def send_gripper_command(self, force_release=False):
        if self.robotiq_gripper is None:
            return

        if force_release:
            await self.robotiq_gripper.automatic_release()
            with self.lock:
                self.target_grip[0] = 0.0
            return

        timeout_exceeded = (
            time.monotonic() - self.gripper_timeout["last_grip"]
        ) * 1000 > self.gripper_timeout["timeout"]

        with self.lock:
            g = float(self.target_grip[0])

        if g > 0.5 and timeout_exceeded:
            await self.robotiq_gripper.automatic_grip()
            with self.lock:
                self.target_grip[0] = 0.0
            self.gripper_timeout["last_grip"] = time.monotonic()

        elif g < -0.5:
            await self.robotiq_gripper.automatic_release()
            with self.lock:
                self.target_grip[0] = 0.0

    async def _go_to_reset_pose(self):
        # Stop torque sending by stopping script/modes before moveJ/moveL
        try:
            self.ur_control.forceModeStop()
        except Exception:
            pass
        try:
            self.ur_control.servoStop()
        except Exception:
            pass

        if self.robotiq_gripper:
            await self.send_gripper_command(force_release=True)
            time.sleep(0.05)

        await self._update_robot_state()

        # Move up a bit first (optional safety)
        try:
            # speedL expects [vx,vy,vz,wx,wy,wz]
            while self.curr_pos[2] < self.reset_height:
                self.ur_control.speedL(
                    [0.0, 0.0, 0.10, 0.0, 0.0, 0.0], acceleration=0.4
                )
                await self._update_robot_state()
                time.sleep(0.01)
            self.ur_control.speedStop(a=0.8)
        except Exception:
            pass

        # Choose reset path
        with self.lock:
            use_pose = np.std(self.reset_pose) > 1e-4
            reset_pose = self.reset_pose.copy()
            reset_q = self.reset_Q.copy()
            self.reset_pose[:] = 0.0

        ok = True
        try:
            if use_pose:
                ok = ok and self.ur_control.moveL(
                    reset_pose.tolist(), speed=0.3, acceleration=0.2
                )
            else:
                ok = ok and self.ur_control.moveJ(
                    reset_q.tolist(), speed=0.5, acceleration=0.3
                )
        except Exception:
            ok = False

        await self._update_robot_state()
        with self.lock:
            self.target_pos[:] = self.curr_pos.copy()
            self._last_target_time = time.monotonic()

        if ok:
            self._reset.clear()
        else:
            self._is_truncated.set()

    # ------------------------- Main loop -------------------------

    def run(self):
        try:
            asyncio.run(self.run_async())
        finally:
            self.stop()

    async def run_async(self):
        await self.start_ur_interfaces(gripper=True)

        # Initialize state + target
        await self._update_robot_state()
        with self.lock:
            self.target_pos[:] = self.curr_pos.copy()
            self._last_target_time = time.monotonic()

        self._is_ready.set()
        if self.verbose:
            print("[TORQUE] Controller ready.")

        try:
            while not self.stopped():
                if self._reset.is_set():
                    await self._go_to_reset_pose()

                t0 = time.monotonic()

                await self._update_robot_state()

                # watchdog: if no target updates, go limp (zero torques)
                if (time.monotonic() - self._last_target_time) > self.watchdog_s:
                    tau = np.zeros((6,), dtype=np.float32)
                else:
                    # compute task-space wrench
                    target = self.get_target_pos(copy=True)
                    curr = self.curr_pos.copy()

                    ep, er = self._pose_error(target, curr)
                    ep, er = self._clip_pose_error(ep, er)

                    v = self.curr_vel[:3]
                    w = self.curr_vel[3:]

                    # wrench = [Fx,Fy,Fz, Tx,Ty,Tz]
                    F = self.Kp_pos * ep - self.Kd_pos * v
                    T = self.Kp_rot * er - self.Kd_rot * w
                    wrench = np.concatenate([F, T], axis=0)  # (6,)

                    # Jacobian -> joint torques
                    J = self._get_jacobian()
                    if J is None:
                        # no Jacobian API available -> cannot do torque mapping robustly
                        self._is_truncated.set()
                        tau = np.zeros((6,), dtype=np.float32)
                    else:
                        tau = (J.T @ wrench).astype(np.float32)

                        # add joint damping
                        tau -= self.Kq * self.curr_Qd

                # clamp & rate limit
                tau = np.clip(tau, -self.tau_limit, self.tau_limit)
                tau = self._rate_limit_tau(tau)

                # send torques
                ok = self._send_joint_torques(tau)
                if not ok:
                    self._is_truncated.set()
                    # attempt to stop script to release control
                    try:
                        self.ur_control.stopScript()
                    except Exception:
                        pass

                # gripper update
                if self.robotiq_gripper:
                    await self.send_gripper_command()

                # maintain loop period
                dt = time.monotonic() - t0
                to_sleep = max(0.0, self.dt - dt)
                time.sleep(to_sleep)

        finally:
            # Always cleanup so you don't get "another thread is already controlling robot"
            try:
                # Best-effort: stop anything running
                try:
                    self.ur_control.forceModeStop()
                except Exception:
                    pass
                try:
                    self.ur_control.servoStop()
                except Exception:
                    pass
                try:
                    self.ur_control.stopScript()
                except Exception:
                    pass
            finally:
                try:
                    if self.robotiq_gripper:
                        await self.send_gripper_command(force_release=True)
                        await self.robotiq_gripper.disconnect()
                except Exception:
                    pass
                try:
                    self.ur_control.disconnect()
                except Exception:
                    pass
                try:
                    self.ur_receive.disconnect()
                except Exception:
                    pass

            if self.verbose:
                print("[TORQUE] Disconnected.")
