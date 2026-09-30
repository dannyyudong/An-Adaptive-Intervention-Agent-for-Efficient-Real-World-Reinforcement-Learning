"""Common lifecycle interfaces for high-level SMDP options.

This module deliberately contains no task-specific policy loading or robot logic.
Concrete options receive their policy/controller through callables so that the
actor loop can use all options through the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Callable, Mapping, Optional

import gymnasium as gym
import numpy as np

from serl_launcher.aia.code_policy import (
    PlanProvider,
    PrimitivePlan,
    PrimitiveStage,
    PrimitiveType,
)


Observation = Any
PolicyFn = Callable[[Observation], np.ndarray]
AvailabilityFn = Callable[[Observation], bool]
ConfidenceFn = Callable[[Observation], float]
TargetProvider = Callable[[Observation], Optional[Any]]
CorrectionPolicyFn = Callable[[Observation, Any], np.ndarray]
TargetReachedFn = Callable[[Observation, Any], bool]
TrackingErrorFn = Callable[[Observation, Any], bool]
PrimitiveActionFn = Callable[[Observation, PrimitiveStage], np.ndarray]
StageReachedFn = Callable[[Observation, PrimitiveStage], bool]


class OptionID(IntEnum):
    """Discrete actions exposed to the high-level scheduler."""

    RL = 0
    TRAJECTORY_CORRECTION = 1
    CODE_POLICY = 2


class TerminationReason(str, Enum):
    """Why an option stopped executing."""

    HORIZON_REACHED = "horizon_reached"
    TARGET_REACHED = "target_reached"
    NO_VALID_TARGET = "no_valid_target"
    TRACKING_ERROR = "tracking_error"
    STAGNATION = "stagnation"
    COMPLETED = "completed"
    EPISODE_TERMINATED = "episode_terminated"
    EPISODE_TRUNCATED = "episode_truncated"
    HUMAN_INTERVENTION = "human_intervention"
    SAFETY_INTERRUPTION = "safety_interruption"
    POLICY_UNAVAILABLE = "policy_unavailable"
    EXECUTION_ERROR = "execution_error"
    SCHEDULER_SWITCH = "scheduler_switch"


@dataclass(frozen=True)
class OptionResult:
    """Option-level outcome used to construct one SMDP transition."""

    option_id: OptionID
    start_step: int
    duration: int
    discounted_return: float
    termination_reason: TerminationReason
    confidence: float
    available_at_start: bool
    episode_terminated: bool
    episode_truncated: bool


class BaseOption(ABC):
    """Base class that owns lifecycle and option-level return accounting.

    One call to :meth:`observe` corresponds to one action that was actually
    passed through ``env.step``. Consequently, ``duration`` counts executed
    control steps rather than policy inference calls.
    """

    def __init__(
        self,
        option_id: OptionID,
        action_space: gym.spaces.Box,
        *,
        gamma: float = 0.99,
        availability_fn: Optional[AvailabilityFn] = None,
        confidence_fn: Optional[ConfidenceFn] = None,
        clip_actions: bool = False,
    ) -> None:
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(
                "Options currently require a gymnasium.spaces.Box action space"
            )
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")

        self.option_id = OptionID(option_id)
        self.action_space = action_space
        self.gamma = float(gamma)
        self._availability_fn = availability_fn
        self._confidence_fn = confidence_fn
        self._clip_actions = bool(clip_actions)

        self._active = False
        self._awaiting_observation = False
        self._start_step = -1
        self._duration = 0
        self._discounted_return = 0.0
        self._available_at_start = False
        self._confidence_at_start = 0.0
        self._episode_terminated = False
        self._episode_truncated = False
        self._last_observation: Observation = None
        self._last_info: Mapping[str, Any] = {}

    @property
    def active(self) -> bool:
        return self._active

    @property
    def duration(self) -> int:
        return self._duration

    @property
    def discounted_return(self) -> float:
        return self._discounted_return

    def available(self, obs: Observation) -> bool:
        if self._availability_fn is None:
            return True
        return bool(self._availability_fn(obs))

    def confidence(self, obs: Observation) -> float:
        value = 1.0 if self._confidence_fn is None else float(self._confidence_fn(obs))
        if not np.isfinite(value):
            raise ValueError(f"Option confidence must be finite, got {value}")
        return value

    def start(self, obs: Observation, *, global_step: int) -> None:
        if self._active:
            raise RuntimeError(f"Option {self.option_id.name} is already active")

        available = self.available(obs)
        if not available:
            raise RuntimeError(f"Option {self.option_id.name} is unavailable")

        self._active = True
        self._awaiting_observation = False
        self._start_step = int(global_step)
        self._duration = 0
        self._discounted_return = 0.0
        self._available_at_start = available
        self._confidence_at_start = self.confidence(obs)
        self._episode_terminated = False
        self._episode_truncated = False
        self._last_observation = obs
        self._last_info = {}
        try:
            self._start_impl(obs)
        except Exception:
            self._active = False
            raise

    def act(self, obs: Observation) -> np.ndarray:
        self._require_active()
        if self._awaiting_observation:
            raise RuntimeError(
                "observe() must be called before requesting the next action"
            )
        if self.should_terminate():
            raise RuntimeError(
                f"Option {self.option_id.name} has reached a termination condition"
            )

        action = self._validate_action(self._act_impl(obs))
        self._awaiting_observation = True
        return action

    def observe(
        self,
        next_obs: Observation,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._require_active()
        if not self._awaiting_observation:
            raise RuntimeError("act() must be called before observe()")

        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError(f"Option reward must be finite, got {reward}")

        self._discounted_return += (self.gamma**self._duration) * reward
        self._duration += 1
        self._episode_terminated = bool(terminated)
        self._episode_truncated = bool(truncated)
        self._last_observation = next_obs
        self._last_info = {} if info is None else info
        self._awaiting_observation = False
        self._observe_impl(next_obs, reward, terminated, truncated, self._last_info)

    def should_terminate(self) -> bool:
        self._require_active()
        if self._episode_terminated or self._episode_truncated:
            return True
        return self._should_terminate_impl()

    def stop(self, reason: Optional[TerminationReason] = None) -> OptionResult:
        self._require_active()
        if self._awaiting_observation:
            raise RuntimeError(
                "Cannot stop an option before observing its executed action"
            )

        resolved_reason = self._resolve_termination_reason(reason)
        result = OptionResult(
            option_id=self.option_id,
            start_step=self._start_step,
            duration=self._duration,
            discounted_return=float(self._discounted_return),
            termination_reason=resolved_reason,
            confidence=float(self._confidence_at_start),
            available_at_start=self._available_at_start,
            episode_terminated=self._episode_terminated,
            episode_truncated=self._episode_truncated,
        )
        try:
            self._stop_impl(resolved_reason)
        finally:
            self._active = False
            self._awaiting_observation = False
        return result

    def _resolve_termination_reason(
        self, reason: Optional[TerminationReason]
    ) -> TerminationReason:
        if reason is not None:
            return TerminationReason(reason)
        if self._episode_terminated:
            return TerminationReason.EPISODE_TERMINATED
        if self._episode_truncated:
            return TerminationReason.EPISODE_TRUNCATED
        inferred = self._termination_reason_impl()
        if inferred is None:
            raise ValueError(
                "A termination reason is required when the option has no internal "
                "termination condition"
            )
        return inferred

    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=self.action_space.dtype)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Option {self.option_id.name} produced action shape {action.shape}; "
                f"expected {self.action_space.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError(
                f"Option {self.option_id.name} produced NaN or Inf action values"
            )

        low = np.asarray(self.action_space.low, dtype=self.action_space.dtype)
        high = np.asarray(self.action_space.high, dtype=self.action_space.dtype)
        if self._clip_actions:
            action = np.clip(action, low, high)
        elif np.any(action < low) or np.any(action > high):
            raise ValueError(
                f"Option {self.option_id.name} produced an out-of-bounds action"
            )
        return action

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError(f"Option {self.option_id.name} is not active")

    def _start_impl(self, obs: Observation) -> None:
        del obs

    @abstractmethod
    def _act_impl(self, obs: Observation) -> np.ndarray:
        """Produce one unvalidated low-level action."""

    def _observe_impl(
        self,
        next_obs: Observation,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> None:
        del next_obs, reward, terminated, truncated, info

    def _should_terminate_impl(self) -> bool:
        return False

    def _termination_reason_impl(self) -> Optional[TerminationReason]:
        return None

    def _stop_impl(self, reason: TerminationReason) -> None:
        del reason


class RLOption(BaseOption):
    """Execute an injected RL policy for at most ``horizon`` control steps."""

    def __init__(
        self,
        policy_fn: PolicyFn,
        action_space: gym.spaces.Box,
        *,
        horizon: int,
        gamma: float = 0.99,
        availability_fn: Optional[AvailabilityFn] = None,
        confidence_fn: Optional[ConfidenceFn] = None,
        clip_actions: bool = False,
    ) -> None:
        if not callable(policy_fn):
            raise TypeError("policy_fn must be callable")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        super().__init__(
            OptionID.RL,
            action_space,
            gamma=gamma,
            availability_fn=availability_fn,
            confidence_fn=confidence_fn,
            clip_actions=clip_actions,
        )
        self.policy_fn = policy_fn
        self.horizon = int(horizon)

    def _act_impl(self, obs: Observation) -> np.ndarray:
        return self.policy_fn(obs)

    def _should_terminate_impl(self) -> bool:
        return self.duration >= self.horizon

    def _termination_reason_impl(self) -> Optional[TerminationReason]:
        if self.duration >= self.horizon:
            return TerminationReason.HORIZON_REACHED
        return None


class TrajectoryCorrectionOption(BaseOption):
    """Closed-loop correction toward a target selected from an expert trajectory.

    The option owns only the common lifecycle. Target selection, observation
    parsing, coordinate transforms, and robot control are injected as callables
    so this class is independent of a particular robot environment.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        *,
        target_provider: TargetProvider,
        correction_policy_fn: CorrectionPolicyFn,
        target_reached_fn: TargetReachedFn,
        max_steps: int,
        gamma: float = 0.99,
        availability_fn: Optional[AvailabilityFn] = None,
        confidence_fn: Optional[ConfidenceFn] = None,
        tracking_error_fn: Optional[TrackingErrorFn] = None,
        clip_actions: bool = False,
    ) -> None:
        if not callable(target_provider):
            raise TypeError("target_provider must be callable")
        if not callable(correction_policy_fn):
            raise TypeError("correction_policy_fn must be callable")
        if not callable(target_reached_fn):
            raise TypeError("target_reached_fn must be callable")
        if tracking_error_fn is not None and not callable(tracking_error_fn):
            raise TypeError("tracking_error_fn must be callable")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")

        super().__init__(
            OptionID.TRAJECTORY_CORRECTION,
            action_space,
            gamma=gamma,
            availability_fn=availability_fn,
            confidence_fn=confidence_fn,
            clip_actions=clip_actions,
        )
        self.target_provider = target_provider
        self.correction_policy_fn = correction_policy_fn
        self.target_reached_fn = target_reached_fn
        self.tracking_error_fn = tracking_error_fn
        self.max_steps = int(max_steps)
        self.target: Optional[Any] = None
        self.target_reached = False
        self.tracking_error = False

    def _start_impl(self, obs: Observation) -> None:
        self.target = self.target_provider(obs)
        if self.target is None:
            raise RuntimeError("No valid trajectory correction target is available")
        self.target_reached = bool(self.target_reached_fn(obs, self.target))
        self.tracking_error = False

    def _act_impl(self, obs: Observation) -> np.ndarray:
        if self.target is None:
            raise RuntimeError("Trajectory correction target has not been initialized")
        return self.correction_policy_fn(obs, self.target)

    def _observe_impl(
        self,
        next_obs: Observation,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> None:
        del reward, terminated, truncated, info
        if self.target is None:
            raise RuntimeError("Trajectory correction target has not been initialized")
        self.target_reached = bool(self.target_reached_fn(next_obs, self.target))
        if self.tracking_error_fn is not None:
            self.tracking_error = bool(self.tracking_error_fn(next_obs, self.target))

    def _should_terminate_impl(self) -> bool:
        return (
            self.target_reached
            or self.tracking_error
            or self.duration >= self.max_steps
        )

    def _termination_reason_impl(self) -> Optional[TerminationReason]:
        if self.target_reached:
            return TerminationReason.TARGET_REACHED
        if self.tracking_error:
            return TerminationReason.TRACKING_ERROR
        if self.duration >= self.max_steps:
            return TerminationReason.HORIZON_REACHED
        return None

    def _stop_impl(self, reason: TerminationReason) -> None:
        del reason
        self.target = None
        self.target_reached = False
        self.tracking_error = False


class CodePolicyOption(BaseOption):
    """Execute a geometry-based primitive plan until completion or interruption."""

    _RESUMABLE_REASONS = frozenset(
        {
            TerminationReason.SCHEDULER_SWITCH,
            TerminationReason.HUMAN_INTERVENTION,
        }
    )

    def __init__(
        self,
        action_space: gym.spaces.Box,
        *,
        plan_provider: PlanProvider,
        primitive_action_fn: PrimitiveActionFn,
        stage_reached_fn: StageReachedFn,
        max_steps: int,
        gamma: float = 0.99,
        clip_actions: bool = False,
    ) -> None:
        if not isinstance(plan_provider, PlanProvider):
            raise TypeError("plan_provider must implement PlanProvider")
        if not callable(primitive_action_fn):
            raise TypeError("primitive_action_fn must be callable")
        if not callable(stage_reached_fn):
            raise TypeError("stage_reached_fn must be callable")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")

        super().__init__(
            OptionID.CODE_POLICY,
            action_space,
            gamma=gamma,
            availability_fn=plan_provider.available,
            confidence_fn=plan_provider.confidence,
            clip_actions=clip_actions,
        )
        self.plan_provider = plan_provider
        self.primitive_action_fn = primitive_action_fn
        self.stage_reached_fn = stage_reached_fn
        self.max_steps = int(max_steps)
        self.plan: Optional[PrimitivePlan] = None
        self.stage_index = 0
        self.stage_steps = 0
        self.completed = False
        self.execution_error = False
        self._resume_plan: Optional[PrimitivePlan] = None
        self._resume_stage_index = 0
        self._resume_stage_steps = 0
        self._resume_reason: Optional[TerminationReason] = None
        self._completed_this_episode = False
        self._resume_rejection_reason: Optional[str] = None

    @property
    def current_stage(self) -> Optional[PrimitiveStage]:
        if self.plan is None or self.stage_index >= len(self.plan.stages):
            return None
        return self.plan.stages[self.stage_index]

    @property
    def resume_pending(self) -> bool:
        return self._resume_plan is not None

    @property
    def completed_this_episode(self) -> bool:
        return self._completed_this_episode

    def available(self, obs: Observation) -> bool:
        if self._completed_this_episode:
            return False
        if not super().available(obs):
            return False
        if self._resume_plan is None:
            self._resume_rejection_reason = None
            return True

        try:
            resume_index = self._reconcile_resume_index(
                obs,
                self._resume_plan,
                self._resume_stage_index,
            )
        except ValueError as exc:
            reason = str(exc)
            if reason != self._resume_rejection_reason:
                print(
                    "[CodePolicy resume] rejected "
                    f"saved_stage={self._stage_name(self._resume_plan, self._resume_stage_index)} "
                    f"reason={reason}",
                    flush=True,
                )
            self._resume_rejection_reason = reason
            return False

        self._resume_rejection_reason = None
        if resume_index >= len(self._resume_plan.stages):
            self._completed_this_episode = True
            self._clear_resume_checkpoint()
            print(
                "[CodePolicy resume] current state already satisfies the remaining plan; "
                "CodePolicy disabled for this episode",
                flush=True,
            )
            return False
        return True

    def reset_episode(self) -> None:
        """Clear all execution checkpoints after an environment/session reset."""
        if self.active:
            raise RuntimeError("Cannot reset CodePolicy while it is active")
        self.plan_provider.reset()
        self._clear_active_execution()
        self._clear_resume_checkpoint()
        self._completed_this_episode = False
        self._resume_rejection_reason = None

    def _start_impl(self, obs: Observation) -> None:
        if self._resume_plan is None:
            self.plan = self.plan_provider.build_plan(obs)
            if self.plan is None:
                raise RuntimeError("CodePolicy could not build a valid primitive plan")
            self.stage_index = self._reconcile_start_index(obs, self.plan)
            self.stage_steps = 0
            if self.stage_index > 0:
                print(
                    "[CodePolicy start] state-aligned "
                    f"stage={self._stage_name(self.plan, self.stage_index)} "
                    f"stage_index={self.stage_index}",
                    flush=True,
                )
        else:
            saved_plan = self._resume_plan
            saved_index = self._resume_stage_index
            saved_steps = self._resume_stage_steps
            saved_reason = self._resume_reason
            resume_index = self._reconcile_resume_index(
                obs,
                saved_plan,
                saved_index,
            )
            if resume_index >= len(saved_plan.stages):
                raise RuntimeError(
                    "CodePolicy resume became complete after availability validation"
                )
            self.plan = saved_plan
            self.stage_index = resume_index
            self.stage_steps = saved_steps if resume_index == saved_index else 0
            self._clear_resume_checkpoint()
            print(
                "[CodePolicy resume] accepted "
                f"reason={saved_reason.value if saved_reason is not None else 'unknown'} "
                f"saved_stage={self._stage_name(saved_plan, saved_index)} "
                f"resume_stage={self._stage_name(saved_plan, resume_index)} "
                f"saved_stage_steps={saved_steps} "
                f"resume_stage_steps={self.stage_steps}",
                flush=True,
            )
        self.completed = False
        self.execution_error = False

    def _reconcile_start_index(
        self,
        obs: Observation,
        plan: PrimitivePlan,
    ) -> int:
        start_index = self.plan_provider.reconcile_start_stage(obs, plan)
        if isinstance(start_index, (bool, np.bool_)):
            raise ValueError("start stage index must be an integer, not bool")
        try:
            start_index = int(start_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("start stage index must be an integer") from exc
        if start_index < 0 or start_index >= len(plan.stages):
            raise ValueError(
                f"start stage {start_index} is outside plan length {len(plan.stages)}"
            )
        return start_index

    def _act_impl(self, obs: Observation) -> np.ndarray:
        stage = self.current_stage
        if stage is None:
            raise RuntimeError("CodePolicy primitive plan has no active stage")
        return self.primitive_action_fn(obs, stage)

    def _observe_impl(
        self,
        next_obs: Observation,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> None:
        del reward, terminated, truncated, info
        stage = self.current_stage
        if stage is None:
            self.completed = True
            return

        self.stage_steps += 1
        if stage.primitive_type is PrimitiveType.MOVE_TO_POSE:
            stage_complete = bool(self.stage_reached_fn(next_obs, stage))
        else:
            stage_complete = self.stage_steps >= stage.min_steps

        if stage_complete:
            self.stage_index += 1
            self.stage_steps = 0
            self.completed = self.plan is not None and self.stage_index >= len(
                self.plan.stages
            )
        elif self.stage_steps >= stage.max_steps:
            self.execution_error = True

    def _should_terminate_impl(self) -> bool:
        return self.completed or self.execution_error or self.duration >= self.max_steps

    def _termination_reason_impl(self) -> Optional[TerminationReason]:
        if self.completed:
            return TerminationReason.COMPLETED
        if self.execution_error:
            return TerminationReason.EXECUTION_ERROR
        if self.duration >= self.max_steps:
            return TerminationReason.HORIZON_REACHED
        return None

    def _stop_impl(self, reason: TerminationReason) -> None:
        if reason in self._RESUMABLE_REASONS and self.plan is not None:
            self._resume_plan = self.plan
            self._resume_stage_index = self.stage_index
            self._resume_stage_steps = self.stage_steps
            self._resume_reason = reason
            self._resume_rejection_reason = None
            print(
                "[CodePolicy resume] checkpoint saved "
                f"reason={reason.value} "
                f"stage={self._stage_name(self.plan, self.stage_index)} "
                f"stage_steps={self.stage_steps}",
                flush=True,
            )
        else:
            self._clear_resume_checkpoint()
            if reason is TerminationReason.COMPLETED:
                self._completed_this_episode = True
        self._clear_active_execution()

    def _reconcile_resume_index(
        self,
        obs: Observation,
        plan: PrimitivePlan,
        saved_index: int,
    ) -> int:
        resume_index = self.plan_provider.reconcile_resume_stage(
            obs,
            plan,
            saved_index,
        )
        if isinstance(resume_index, (bool, np.bool_)):
            raise ValueError("resume stage index must be an integer, not bool")
        try:
            resume_index = int(resume_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("resume stage index must be an integer") from exc
        if resume_index < saved_index:
            raise ValueError(
                f"resume stage moved backward from {saved_index} to {resume_index}"
            )
        if resume_index > len(plan.stages):
            raise ValueError(
                f"resume stage {resume_index} exceeds plan length {len(plan.stages)}"
            )
        return resume_index

    @staticmethod
    def _stage_name(plan: PrimitivePlan, stage_index: int) -> str:
        if stage_index >= len(plan.stages):
            return "completed"
        return plan.stages[stage_index].name

    def _clear_active_execution(self) -> None:
        self.plan = None
        self.stage_index = 0
        self.stage_steps = 0
        self.completed = False
        self.execution_error = False

    def _clear_resume_checkpoint(self) -> None:
        self._resume_plan = None
        self._resume_stage_index = 0
        self._resume_stage_steps = 0
        self._resume_reason = None


class UnavailableOption(BaseOption):
    """Explicit placeholder for a policy backend that is not implemented yet."""

    def __init__(
        self,
        option_id: OptionID,
        action_space: gym.spaces.Box,
        *,
        reason: str,
    ) -> None:
        if OptionID(option_id) is OptionID.RL:
            raise ValueError("Use RLOption for the RL option")
        super().__init__(option_id, action_space)
        self.reason = str(reason)

    def available(self, obs: Observation) -> bool:
        del obs
        return False

    def confidence(self, obs: Observation) -> float:
        del obs
        return 0.0

    def _act_impl(self, obs: Observation) -> np.ndarray:
        del obs
        raise RuntimeError(
            f"Option {self.option_id.name} is unavailable: {self.reason}"
        )
