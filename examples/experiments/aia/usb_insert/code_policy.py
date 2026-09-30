"""Task-specific geometry CodePolicy for deterministic USB insertion."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

from serl_launcher.aia.code_policy import (
    PlanProvider,
    PrimitivePlan,
    PrimitiveStage,
    PrimitiveType,
)


CurrentPoseFn = Callable[[], np.ndarray]


def _raw_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def current_contact_force_z(env: Any) -> float:
    """Read the filtered force-z signal used by the robot controller."""
    controller = getattr(_raw_env(env), "controller", None)
    if controller is None:
        return float("nan")

    for attr in ("curr_force_lowpass", "curr_force"):
        value = getattr(controller, attr, None)
        if value is None:
            continue
        force = np.asarray(value, dtype=np.float64).reshape(-1)
        if force.size >= 3:
            return float(force[2])

    state_getter = getattr(controller, "get_state", None)
    if callable(state_getter):
        try:
            force = np.asarray(
                state_getter().get("force", []), dtype=np.float64
            ).reshape(-1)
            if force.size >= 3:
                return float(force[2])
        except Exception:
            pass
    return float("nan")


class USBInsertionPlanProvider(PlanProvider):
    """Build ``move above target -> insert vertically`` from a fixed xyz target."""

    def __init__(
        self,
        *,
        target_xyz: np.ndarray,
        current_pose_fn: CurrentPoseFn,
        approach_dz: float,
        position_tolerance: float,
        rotation_tolerance: float,
        move_max_steps: int,
        insert_max_steps: int,
        workspace_low: np.ndarray,
        workspace_high: np.ndarray,
        contact_stop_force_z: float,
    ) -> None:
        self.target_xyz = np.asarray(target_xyz, dtype=np.float32).reshape(3)
        self.current_pose_fn = current_pose_fn
        self.approach_dz = float(approach_dz)
        self.position_tolerance = float(position_tolerance)
        self.rotation_tolerance = float(rotation_tolerance)
        self.move_max_steps = int(move_max_steps)
        self.insert_max_steps = int(insert_max_steps)
        self.workspace_low = np.asarray(workspace_low, dtype=np.float32).reshape(-1)[:3]
        self.workspace_high = np.asarray(workspace_high, dtype=np.float32).reshape(-1)[
            :3
        ]
        self.contact_stop_force_z = float(contact_stop_force_z)

        if self.workspace_low.shape != (3,) or self.workspace_high.shape != (3,):
            raise ValueError("workspace bounds must provide at least xyz values")
        if self.approach_dz <= 0.0:
            raise ValueError("approach_dz must be positive")
        if self.move_max_steps <= 0 or self.insert_max_steps <= 0:
            raise ValueError("USB insertion stage limits must be positive")

    @property
    def approach_xyz(self) -> np.ndarray:
        approach = self.target_xyz.copy()
        approach[2] += self.approach_dz
        return approach

    def available(self, obs: Any) -> bool:
        del obs
        if not np.all(np.isfinite(self.target_xyz)):
            return False
        approach = self.approach_xyz
        return bool(
            np.all(self.target_xyz >= self.workspace_low)
            and np.all(self.target_xyz <= self.workspace_high)
            and np.all(approach >= self.workspace_low)
            and np.all(approach <= self.workspace_high)
        )

    def confidence(self, obs: Any) -> float:
        return 1.0 if self.available(obs) else 0.0

    def build_plan(self, obs: Any) -> PrimitivePlan | None:
        if not self.available(obs):
            return None
        current_pose = np.asarray(self.current_pose_fn(), dtype=np.float32).reshape(-1)
        if current_pose.shape != (7,) or not np.all(np.isfinite(current_pose)):
            raise ValueError(
                "current_pose_fn must return finite xyz+quaternion shape (7,)"
            )
        quat_norm = float(np.linalg.norm(current_pose[3:7]))
        if quat_norm <= 1e-8:
            raise ValueError("current TCP pose contains a zero quaternion")
        target_quat = current_pose[3:7] / quat_norm

        common = {
            "target_quat": target_quat,
            "min_steps": 1,
            "position_tolerance": self.position_tolerance,
            "rotation_tolerance": self.rotation_tolerance,
        }
        return PrimitivePlan(
            name="usb_insertion",
            source="fixed_config_target",
            confidence=1.0,
            stages=(
                PrimitiveStage(
                    PrimitiveType.MOVE_TO_POSE,
                    "move_above_usb_target",
                    target_xyz=self.approach_xyz,
                    max_steps=self.move_max_steps,
                    metadata={"motion": "approach", "contact_stop": False},
                    **common,
                ),
                PrimitiveStage(
                    PrimitiveType.MOVE_TO_POSE,
                    "insert_usb_downward",
                    target_xyz=self.target_xyz,
                    max_steps=self.insert_max_steps,
                    metadata={
                        "motion": "insertion",
                        "contact_stop": True,
                        "contact_stop_force_z": self.contact_stop_force_z,
                    },
                    **common,
                ),
            ),
            metadata={
                "target_xyz": self.target_xyz.tolist(),
                "approach_xyz": self.approach_xyz.tolist(),
            },
        )


def usb_stage_reached(env: Any, stage: PrimitiveStage) -> bool:
    """Check pose convergence, with force-z contact stop during insertion."""
    raw = _raw_env(env)
    update_pose = getattr(raw, "_update_currpos", None)
    if callable(update_pose):
        update_pose()
    current_pose = np.asarray(getattr(raw, "curr_pos", []), dtype=np.float32).reshape(
        -1
    )
    if current_pose.shape != (7,) or stage.target_xyz is None:
        raise RuntimeError("USB environment does not expose a valid current TCP pose")

    position_error = float(np.linalg.norm(current_pose[:3] - stage.target_xyz))
    if stage.target_quat is None:
        rotation_error = 0.0
    else:
        rotation_error = float(
            (
                R.from_quat(stage.target_quat) * R.from_quat(current_pose[3:7]).inv()
            ).magnitude()
        )
    pose_reached = bool(
        position_error <= stage.position_tolerance
        and rotation_error <= stage.rotation_tolerance
    )

    force_z = float("nan")
    contact_reached = False
    if bool(stage.metadata.get("contact_stop", False)):
        threshold = float(stage.metadata.get("contact_stop_force_z", float("inf")))
        force_z = current_contact_force_z(env)
        contact_reached = bool(np.isfinite(force_z) and force_z >= threshold)

    print(
        "[CodePolicy reach] "
        f"stage={stage.name} "
        f"position_error={position_error:.6f}m "
        f"rotation_error={rotation_error:.6f}rad "
        f"force_z={force_z:.4f}N "
        f"pose_reached={pose_reached} contact_reached={contact_reached}",
        flush=True,
    )

    if contact_reached:
        holder = getattr(raw, "hold_position", None)
        if callable(holder):
            holder()
    return pose_reached or contact_reached
