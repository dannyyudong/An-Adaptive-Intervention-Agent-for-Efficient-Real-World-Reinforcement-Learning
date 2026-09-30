"""SMDP transition construction for the high-level Scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from serl_launcher.aia.options import OptionID, OptionResult, TerminationReason
from serl_launcher.aia.state import SCHEDULER_STATE_SCHEMA_VERSION


SCHEDULER_TRANSITION_SCHEMA_VERSION = "scheduler_smdp_transition_v2"
TERMINATION_REASON_TO_ID = {
    TerminationReason.HORIZON_REACHED: 0,
    TerminationReason.TARGET_REACHED: 1,
    TerminationReason.NO_VALID_TARGET: 2,
    TerminationReason.TRACKING_ERROR: 3,
    TerminationReason.STAGNATION: 4,
    TerminationReason.COMPLETED: 5,
    TerminationReason.EPISODE_TERMINATED: 6,
    TerminationReason.EPISODE_TRUNCATED: 7,
    TerminationReason.HUMAN_INTERVENTION: 8,
    TerminationReason.SAFETY_INTERRUPTION: 9,
    TerminationReason.POLICY_UNAVAILABLE: 10,
    TerminationReason.EXECUTION_ERROR: 11,
    TerminationReason.SCHEDULER_SWITCH: 12,
}


@dataclass(frozen=True)
class SchedulerRewardConfig:
    gamma: float = 0.99
    trajectory_cost: float = 0.02
    code_policy_cost: float = 0.05
    duration_cost: float = 0.01
    max_duration: int = 200

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.gamma) <= 1.0:
            raise ValueError("Scheduler gamma must be in [0, 1]")
        if int(self.max_duration) <= 0:
            raise ValueError("Scheduler max_duration must be positive")
        for name in (
            "trajectory_cost",
            "code_policy_cost",
            "duration_cost",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def reward(
        self,
        result: OptionResult,
        *,
        task_return_override: Optional[float] = None,
    ) -> float:
        task_return = (
            float(result.discounted_return)
            if task_return_override is None
            else float(task_return_override)
        )
        value = task_return
        if result.option_id is OptionID.TRAJECTORY_CORRECTION:
            value -= float(self.trajectory_cost)
        elif result.option_id is OptionID.CODE_POLICY:
            value -= float(self.code_policy_cost)
        value -= float(self.duration_cost) * min(
            float(result.duration) / float(self.max_duration), 1.0
        )
        if not np.isfinite(value):
            raise ValueError("Scheduler reward must be finite")
        return float(value)


@dataclass(frozen=True)
class PendingSchedulerTransition:
    observation: np.ndarray
    option_id: OptionID
    available_actions: np.ndarray
    behavior_prob: float
    start_step: int
    rl_policy_version: int


class SchedulerTransitionBuilder:
    def __init__(
        self,
        state_dim: int,
        reward_config: SchedulerRewardConfig,
        *,
        state_schema_version: str = SCHEDULER_STATE_SCHEMA_VERSION,
    ) -> None:
        if int(state_dim) <= 0:
            raise ValueError("Scheduler state_dim must be positive")
        if not str(state_schema_version):
            raise ValueError("state_schema_version must be non-empty")
        self.state_dim = int(state_dim)
        self.reward_config = reward_config
        self.state_schema_version = str(state_schema_version)

    @staticmethod
    def _validate_action_mask(mask: Any, *, allow_empty: bool) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(OptionID),):
            raise ValueError(
                f"Scheduler action mask must have shape {(len(OptionID),)}, "
                f"got {mask.shape}"
            )
        if not allow_empty and not np.any(mask):
            raise ValueError("Scheduler action mask must contain a valid action")
        return mask

    def begin(
        self,
        observation: np.ndarray,
        option_id: OptionID,
        available_actions: np.ndarray,
        *,
        behavior_prob: float,
        start_step: int,
        rl_policy_version: int = 0,
    ) -> PendingSchedulerTransition:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.state_dim,) or not np.all(
            np.isfinite(observation)
        ):
            raise ValueError(
                f"Scheduler observation must have shape {(self.state_dim,)}"
            )
        behavior_prob = float(behavior_prob)
        if not np.isfinite(behavior_prob) or not 0.0 < behavior_prob <= 1.0:
            raise ValueError("behavior_prob must be in (0, 1]")
        option_id = OptionID(option_id)
        action_mask = self._validate_action_mask(available_actions, allow_empty=False)
        if not action_mask[int(option_id)]:
            raise ValueError(f"Selected Option {option_id.name} is masked out")
        return PendingSchedulerTransition(
            observation=observation.copy(),
            option_id=option_id,
            available_actions=action_mask.copy(),
            behavior_prob=behavior_prob,
            start_step=int(start_step),
            rl_policy_version=int(rl_policy_version),
        )

    def finish(
        self,
        pending: PendingSchedulerTransition,
        result: OptionResult,
        next_observation: np.ndarray,
        next_available_actions: np.ndarray,
        *,
        scheduler_reward: float,
        terminated_override: Optional[bool] = None,
        truncated_override: Optional[bool] = None,
    ) -> Mapping[str, Any]:
        if result.duration <= 0:
            raise ValueError("Cannot create a Scheduler transition with zero duration")
        if result.option_id is not pending.option_id:
            raise ValueError("Pending and completed Option ids do not match")
        next_observation = np.asarray(next_observation, dtype=np.float32)
        if next_observation.shape != (self.state_dim,) or not np.all(
            np.isfinite(next_observation)
        ):
            raise ValueError(
                f"Scheduler next observation must have shape {(self.state_dim,)}"
            )
        terminated = (
            bool(result.episode_terminated)
            if terminated_override is None
            else bool(terminated_override)
        )
        truncated = (
            bool(result.episode_truncated)
            if truncated_override is None
            else bool(truncated_override)
        )
        done = bool(terminated or truncated)
        next_mask = self._validate_action_mask(next_available_actions, allow_empty=done)
        scheduler_reward = float(scheduler_reward)
        if not np.isfinite(scheduler_reward):
            raise ValueError("Scheduler reward must be finite")
        return {
            "observations": pending.observation.copy(),
            "actions": np.int32(pending.option_id),
            "rewards": np.float32(scheduler_reward),
            "next_observations": next_observation.copy(),
            "durations": np.int32(result.duration),
            "discounts": np.float32(
                float(self.reward_config.gamma) ** int(result.duration)
            ),
            "masks": np.float32(1.0 - float(done)),
            "dones": np.bool_(done),
            "terminated": np.bool_(terminated),
            "truncated": np.bool_(truncated),
            "termination_reason": np.int32(
                TERMINATION_REASON_TO_ID[result.termination_reason]
            ),
            "behavior_prob": np.float32(pending.behavior_prob),
            "available_actions": pending.available_actions.copy(),
            "next_available_actions": next_mask.copy(),
            "start_step": np.int64(pending.start_step),
            "rl_policy_version": np.int64(pending.rl_policy_version),
            "schema_version": SCHEDULER_TRANSITION_SCHEMA_VERSION,
            "state_schema_version": self.state_schema_version,
        }
