"""Geometry-based action primitives used by :class:`CodePolicyOption`.

Perception is intentionally outside this module. In the current deployment the
task environment sends RGB-D plus camera intrinsics to the perception service,
which returns source/target positions in world coordinates. This module turns
those task-specific results into a robot-independent primitive plan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional

import numpy as np


Observation = Any


class PrimitiveType(str, Enum):
    MOVE_TO_POSE = "move_to_pose"
    SET_GRIPPER = "set_gripper"
    HOLD = "hold"


def _optional_vector(value: Any, size: int, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


@dataclass(frozen=True)
class PrimitiveStage:
    """One closed-loop motion, gripper command, or hold stage."""

    primitive_type: PrimitiveType
    name: str
    target_xyz: Optional[np.ndarray] = None
    target_quat: Optional[np.ndarray] = None
    gripper_action: float = 0.0
    min_steps: int = 1
    max_steps: int = 80
    position_tolerance: float = 0.015
    rotation_tolerance: float = 0.08
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "primitive_type", PrimitiveType(self.primitive_type))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(
            self, "target_xyz", _optional_vector(self.target_xyz, 3, "target_xyz")
        )
        target_quat = _optional_vector(self.target_quat, 4, "target_quat")
        if target_quat is not None:
            norm = float(np.linalg.norm(target_quat))
            if norm <= 1e-8:
                raise ValueError("target_quat contains a zero quaternion")
            target_quat = target_quat / norm
        object.__setattr__(self, "target_quat", target_quat)

        if self.min_steps <= 0:
            raise ValueError("min_steps must be positive")
        if self.max_steps < self.min_steps:
            raise ValueError("max_steps must be greater than or equal to min_steps")
        if self.position_tolerance < 0.0 or self.rotation_tolerance < 0.0:
            raise ValueError("stage tolerances must be non-negative")
        if not np.isfinite(self.gripper_action):
            raise ValueError("gripper_action must be finite")
        if (
            self.primitive_type is PrimitiveType.MOVE_TO_POSE
            and self.target_xyz is None
        ):
            raise ValueError("MOVE_TO_POSE requires target_xyz")


@dataclass(frozen=True)
class PrimitivePlan:
    """An ordered sequence of geometry-based action primitives."""

    name: str
    stages: tuple[PrimitiveStage, ...]
    confidence: float = 1.0
    source: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages:
            raise ValueError("PrimitivePlan must contain at least one stage")
        if not all(isinstance(stage, PrimitiveStage) for stage in stages):
            raise TypeError("PrimitivePlan stages must be PrimitiveStage instances")
        confidence = float(self.confidence)
        if not np.isfinite(confidence):
            raise ValueError("PrimitivePlan confidence must be finite")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source", str(self.source))


class PlanProvider(ABC):
    """Task-specific interface for producing one primitive plan."""

    def available(self, obs: Observation) -> bool:
        del obs
        return True

    def confidence(self, obs: Observation) -> float:
        del obs
        return 1.0

    @abstractmethod
    def build_plan(self, obs: Observation) -> Optional[PrimitivePlan]:
        """Build a plan from current perception or configured target coordinates."""

    def reconcile_start_stage(
        self,
        obs: Observation,
        plan: PrimitivePlan,
    ) -> int:
        """Select the stage for a newly built plan from unambiguous task state.

        This hook is used when CodePolicy has no saved same-episode checkpoint,
        for example when another policy has already completed part of the task.
        Implementations should return stage zero whenever the observed task
        state is missing, contradictory, or insufficient to infer progress.
        """
        del obs, plan
        return 0

    def reconcile_resume_stage(
        self,
        obs: Observation,
        plan: PrimitivePlan,
        stage_index: int,
    ) -> int:
        """Validate and optionally advance a same-episode resume checkpoint.

        Implementations may use current robot, gripper, or task state to skip
        stages that are already satisfied.  The returned index must never move
        backward; returning ``len(plan.stages)`` marks the plan complete.  Raise
        ``ValueError`` when the current state contradicts the saved checkpoint
        and resuming would be unsafe.

        The default is deliberately conservative and keeps the saved stage.
        """
        del obs, plan
        return int(stage_index)

    def reset(self) -> None:
        """Clear episode-specific cached perception or plans."""


class CallablePlanProvider(PlanProvider):
    """Adapt an environment/task callback to the :class:`PlanProvider` API."""

    def __init__(
        self,
        plan_fn: Callable[[Observation], Optional[PrimitivePlan | Mapping[str, Any]]],
        *,
        availability_fn: Optional[Callable[[Observation], bool]] = None,
        confidence_fn: Optional[Callable[[Observation], float]] = None,
        mapping_converter: Optional[
            Callable[[Mapping[str, Any]], PrimitivePlan]
        ] = None,
        reset_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        if not callable(plan_fn):
            raise TypeError("plan_fn must be callable")
        self.plan_fn = plan_fn
        self.availability_fn = availability_fn
        self.confidence_fn = confidence_fn
        self.mapping_converter = mapping_converter or primitive_plan_from_mapping
        self.reset_fn = reset_fn

    def available(self, obs: Observation) -> bool:
        if self.availability_fn is None:
            return True
        return bool(self.availability_fn(obs))

    def confidence(self, obs: Observation) -> float:
        value = 1.0 if self.confidence_fn is None else float(self.confidence_fn(obs))
        if not np.isfinite(value):
            raise ValueError(f"Plan confidence must be finite, got {value}")
        return value

    def build_plan(self, obs: Observation) -> Optional[PrimitivePlan]:
        plan = self.plan_fn(obs)
        if plan is None or isinstance(plan, PrimitivePlan):
            return plan
        if not isinstance(plan, Mapping):
            raise TypeError("plan_fn must return PrimitivePlan, a mapping, or None")
        return self.mapping_converter(plan)

    def reset(self) -> None:
        if self.reset_fn is not None:
            self.reset_fn()


def primitive_plan_from_mapping(plan: Mapping[str, Any]) -> PrimitivePlan:
    """Convert the existing ``plan_intervention_primitive`` mapping format.

    This converter consumes already computed world-frame waypoints. It does not
    perform mask processing, camera deprojection, or coordinate calibration.
    """

    envstep = plan.get("envstep", {}) or {}
    waypoints = plan.get("waypoints", {}) or {}
    safety = plan.get("safety", {}) or {}
    max_steps = max(1, int(envstep.get("max_steps_per_waypoint", 80)))
    gripper_steps = max(1, int(envstep.get("gripper_steps", 10)))
    position_tolerance = float(envstep.get("xyz_tolerance", 0.015))
    rotation_tolerance = float(envstep.get("rot_tolerance", 0.08))
    target_quat = plan.get("home_quat")

    required_waypoints = ("pick_approach", "pick", "place_approach", "place")
    missing = [name for name in required_waypoints if name not in waypoints]
    if missing:
        raise ValueError(f"Primitive plan is missing waypoints: {missing}")

    stages: list[PrimitiveStage] = []
    threshold = float(safety.get("open_gripper_threshold", -1.0))
    closed_norm = float(safety.get("closed_norm", float("nan")))
    if threshold >= 0.0 and (not np.isfinite(closed_norm) or closed_norm > threshold):
        stages.append(
            PrimitiveStage(
                PrimitiveType.SET_GRIPPER,
                "pre_open",
                gripper_action=-1.0,
                min_steps=gripper_steps,
                max_steps=gripper_steps,
            )
        )

    def move(name: str, waypoint: str) -> PrimitiveStage:
        return PrimitiveStage(
            PrimitiveType.MOVE_TO_POSE,
            name,
            target_xyz=waypoints[waypoint],
            target_quat=target_quat,
            min_steps=1,
            max_steps=max_steps,
            position_tolerance=position_tolerance,
            rotation_tolerance=rotation_tolerance,
        )

    stages.extend(
        [
            move("pick_approach", "pick_approach"),
            move("pick", "pick"),
            PrimitiveStage(
                PrimitiveType.SET_GRIPPER,
                "close",
                gripper_action=1.0,
                min_steps=gripper_steps,
                max_steps=gripper_steps,
            ),
            move("lift_after_pick", "pick_approach"),
            move("place_approach", "place_approach"),
            move("place", "place"),
            PrimitiveStage(
                PrimitiveType.SET_GRIPPER,
                "open",
                gripper_action=-1.0,
                min_steps=gripper_steps,
                max_steps=gripper_steps,
            ),
            move("retreat_after_place", "place_approach"),
        ]
    )

    metadata = {
        key: plan[key]
        for key in (
            "source_xyz",
            "target_xyz",
            "place_xyz",
            "placement_mode",
            "primitive_context",
        )
        if key in plan
    }
    return PrimitivePlan(
        name=str(plan.get("primitive", "pick_and_place")),
        stages=tuple(stages),
        confidence=float(plan.get("confidence", 1.0)),
        source=str(plan.get("source", "environment_perception")),
        metadata=metadata,
    )
