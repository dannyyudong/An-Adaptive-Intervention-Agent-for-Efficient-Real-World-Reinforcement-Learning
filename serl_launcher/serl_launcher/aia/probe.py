"""Training-time autonomous RL probes for the high-level Scheduler."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional

import numpy as np

from serl_launcher.aia.trajectory_correct import TrajectoryProgress


RL_PROBE_STATE_SCHEMA_VERSION = "adaptive_rl_probe_v1"


@dataclass(frozen=True)
class RLProbeConfig:
    """Task-configurable autonomous-window curriculum."""

    initial_steps: int = 5
    step_increment: int = 5
    max_steps: int = 50
    required_passes: int = 2
    episode_interval: int = 1
    # Retained for configuration/state compatibility.  Probe qualification now
    # uses local expert-relative motion in metres instead of a trajectory-global
    # normalized floor.
    min_progress_delta: float = 0.01
    expert_progress_fraction: float = 0.8
    reference_motion_floor_m: float = 0.001
    stationary_path_tolerance_m: float = 0.005
    max_path_deviation: float = 0.08
    stall_steps: int = 10
    progress_epsilon: float = 1e-4
    progress_epsilon_m: float = 0.0001
    step_quantum: int = 5
    off_path_decrement: int = 5
    safety_decrement: int = 10

    def __post_init__(self) -> None:
        integer_fields = (
            "initial_steps",
            "step_increment",
            "max_steps",
            "required_passes",
            "episode_interval",
            "stall_steps",
            "step_quantum",
            "off_path_decrement",
            "safety_decrement",
        )
        for name in integer_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.initial_steps > self.max_steps:
            raise ValueError("initial_steps must not exceed max_steps")
        for name in (
            "initial_steps",
            "step_increment",
            "max_steps",
            "off_path_decrement",
            "safety_decrement",
        ):
            if int(getattr(self, name)) % int(self.step_quantum) != 0:
                raise ValueError(f"{name} must be a multiple of step_quantum")
        if (
            not np.isfinite(self.min_progress_delta)
            or not 0.0 <= float(self.min_progress_delta) <= 1.0
        ):
            raise ValueError("min_progress_delta must be in [0, 1]")
        if (
            not np.isfinite(self.expert_progress_fraction)
            or not 0.0 < float(self.expert_progress_fraction) <= 1.0
        ):
            raise ValueError("expert_progress_fraction must be in (0, 1]")
        if (
            not np.isfinite(self.reference_motion_floor_m)
            or float(self.reference_motion_floor_m) < 0.0
        ):
            raise ValueError("reference_motion_floor_m must be non-negative")
        if (
            not np.isfinite(self.stationary_path_tolerance_m)
            or float(self.stationary_path_tolerance_m) <= 0.0
        ):
            raise ValueError("stationary_path_tolerance_m must be positive")
        if (
            not np.isfinite(self.max_path_deviation)
            or float(self.max_path_deviation) <= 0.0
        ):
            raise ValueError("max_path_deviation must be positive")
        if not np.isfinite(self.progress_epsilon) or float(self.progress_epsilon) < 0.0:
            raise ValueError("progress_epsilon must be non-negative")
        if (
            not np.isfinite(self.progress_epsilon_m)
            or float(self.progress_epsilon_m) < 0.0
        ):
            raise ValueError("progress_epsilon_m must be non-negative")


@dataclass(frozen=True)
class RLProbeResult:
    reason: str
    budget_before: int
    budget_after: int
    executed_steps: int
    start_index: int
    end_index: int
    expert_target_index: int
    start_progress: float
    end_progress: float
    expert_target_progress: float
    expert_progress_delta: float
    required_progress_delta: float
    progress_delta: float
    progress_ratio: float
    start_arc_length_m: float
    end_arc_length_m: float
    expert_target_arc_length_m: float
    expert_motion_m: float
    required_motion_m: float
    actual_motion_m: float
    grading_mode: str
    stationary_path_limit_m: float
    max_path_deviation: float
    success: bool
    passed: bool
    pass_streak: int

    def metrics(self) -> dict[str, float | int]:
        budget_delta = int(self.budget_after - self.budget_before)
        return {
            "rl_probe/budget_steps": self.budget_before,
            "rl_probe/next_budget_steps": self.budget_after,
            "rl_probe/current_budget": self.budget_after,
            "rl_probe/budget_delta": budget_delta,
            "rl_probe/budget_grew": int(budget_delta > 0),
            "rl_probe/budget_shrank": int(budget_delta < 0),
            "rl_probe/executed_steps": self.executed_steps,
            "rl_probe/start_index": self.start_index,
            "rl_probe/end_index": self.end_index,
            "rl_probe/expert_target_index": self.expert_target_index,
            "rl_probe/start_progress": self.start_progress,
            "rl_probe/end_progress": self.end_progress,
            "rl_probe/expert_target_progress": self.expert_target_progress,
            "rl_probe/expert_progress_delta": self.expert_progress_delta,
            "rl_probe/required_progress_delta": self.required_progress_delta,
            "rl_probe/progress_delta": self.progress_delta,
            "rl_probe/progress_ratio": self.progress_ratio,
            "rl_probe/start_arc_length_m": self.start_arc_length_m,
            "rl_probe/end_arc_length_m": self.end_arc_length_m,
            "rl_probe/expert_target_arc_length_m": (self.expert_target_arc_length_m),
            "rl_probe/expert_motion_m": self.expert_motion_m,
            "rl_probe/required_motion_m": self.required_motion_m,
            "rl_probe/actual_motion_m": self.actual_motion_m,
            "rl_probe/grading_mode_moving": int(self.grading_mode == "moving"),
            "rl_probe/grading_mode_stationary": int(self.grading_mode == "stationary"),
            "rl_probe/stationary_path_limit_m": self.stationary_path_limit_m,
            "rl_probe/max_path_deviation": self.max_path_deviation,
            "rl_probe/success": int(self.success),
            "rl_probe/pass": int(self.passed),
            "rl_probe/pass_streak": self.pass_streak,
        }


class AdaptiveRLProbeController:
    """Grow an RL-only behavior window from measured expert-path progress.

    This controller chooses behavior only.  Callers must keep the physical
    Scheduler availability mask unchanged in replay transitions.
    """

    def __init__(self, config: RLProbeConfig) -> None:
        self.config = config
        self.budget_steps = int(config.initial_steps)
        self.pass_streak = 0
        self.completed_episodes = 0
        self.scheduled_this_episode = False
        self.active = False
        self.attempted_this_episode = False
        self._episode_open = False
        self.executed_steps = 0
        self.start_index = 0
        self.end_index = 0
        self.expert_target_index = 0
        self.start_progress = 0.0
        self.end_progress = 0.0
        self.expert_target_progress = 0.0
        self.expert_progress_delta = 0.0
        self.required_progress_delta = 0.0
        self.start_arc_length_m = 0.0
        self.end_arc_length_m = 0.0
        self.expert_target_arc_length_m = 0.0
        self.expert_motion_m = 0.0
        self.required_motion_m = 0.0
        self.grading_mode = "stationary"
        self.stationary_path_limit_m = float(config.stationary_path_tolerance_m)
        self.max_path_deviation = 0.0
        self._best_progress = 0.0
        self._best_arc_length_m = 0.0
        self._steps_without_progress = 0

    @property
    def remaining_steps(self) -> int:
        if not self.active:
            return 0
        return max(0, self.budget_steps - self.executed_steps)

    @property
    def episode_counter(self) -> int:
        """One-based ordinal of the current or next Scheduler episode."""

        return int(self.completed_episodes + 1)

    @property
    def episodes_until_next_probe(self) -> int:
        """Number of non-probe episodes before the next scheduled probe."""

        position = self.completed_episodes % self.config.episode_interval
        return int(
            (self.config.episode_interval - position - 1) % self.config.episode_interval
        )

    def schedule_metrics(self) -> dict[str, int]:
        return {
            "rl_probe/episode_counter": self.episode_counter,
            "rl_probe/scheduled_this_episode": int(self.scheduled_this_episode),
            "rl_probe/episodes_until_next": self.episodes_until_next_probe,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": RL_PROBE_STATE_SCHEMA_VERSION,
            "budget_steps": int(self.budget_steps),
            "pass_streak": int(self.pass_streak),
            "completed_episodes": int(self.completed_episodes),
            "config": asdict(self.config),
        }

    def restore_state(self, state: Mapping[str, object]) -> None:
        if state.get("schema_version") != RL_PROBE_STATE_SCHEMA_VERSION:
            raise ValueError("RL probe state schema version mismatch")
        budget_steps = int(state["budget_steps"])
        pass_streak = int(state["pass_streak"])
        completed_episodes = int(state.get("completed_episodes", 0))
        if (
            budget_steps < self.config.initial_steps
            or budget_steps > self.config.max_steps
            or budget_steps % self.config.step_quantum != 0
        ):
            raise ValueError("Saved RL probe budget is incompatible with config")
        if not 0 <= pass_streak < self.config.required_passes:
            raise ValueError("Saved RL probe pass streak is invalid")
        if completed_episodes < 0:
            raise ValueError("Saved RL probe completed_episodes is invalid")
        self.budget_steps = budget_steps
        self.pass_streak = pass_streak
        self.completed_episodes = completed_episodes

    def reset_episode(self) -> bool:
        if self.active:
            raise RuntimeError("Cannot reset an episode during an active RL probe")
        if self._episode_open:
            raise RuntimeError("Complete the current episode before starting another")
        self.scheduled_this_episode = bool(
            self.completed_episodes % self.config.episode_interval == 0
        )
        self.attempted_this_episode = not self.scheduled_this_episode
        self.executed_steps = 0
        self._episode_open = True
        return self.scheduled_this_episode

    def mark_episode_completed(self) -> bool:
        """Advance the interval only after a real environment episode ends."""

        if self.active:
            raise RuntimeError("Finish the active RL probe before the episode")
        if not self._episode_open:
            return False
        self.completed_episodes += 1
        self.scheduled_this_episode = False
        self._episode_open = False
        return True

    def cancel_episode(self) -> None:
        """Prevent a pending probe after manual or runtime control takes over."""
        if self.active:
            raise RuntimeError(
                "Finish an active RL probe before cancelling its episode"
            )
        self.attempted_this_episode = True

    def start_episode(
        self,
        progress: TrajectoryProgress,
        *,
        expert_target_index: int,
        expert_target_progress: float,
        expert_target_arc_length_m: Optional[float] = None,
    ) -> bool:
        if self.active:
            raise RuntimeError("Cannot start a new episode during an active RL probe")
        if self.attempted_this_episode:
            return False
        expert_target_index = int(expert_target_index)
        expert_target_progress = float(expert_target_progress)
        if expert_target_index < int(progress.index):
            raise ValueError("expert_target_index must not precede the start index")
        if (
            not np.isfinite(expert_target_progress)
            or expert_target_progress < float(progress.progress)
            or expert_target_progress > 1.0
        ):
            raise ValueError(
                "expert_target_progress must be finite and in " "[start progress, 1]"
            )
        self.start_index = int(progress.index)
        self.end_index = int(progress.index)
        self.expert_target_index = expert_target_index
        self.start_progress = float(progress.progress)
        self.end_progress = float(progress.progress)
        self.expert_target_progress = expert_target_progress
        self.expert_progress_delta = max(
            0.0,
            self.expert_target_progress - self.start_progress,
        )
        self.start_arc_length_m = float(progress.arc_length_m)
        self.end_arc_length_m = float(progress.arc_length_m)
        if expert_target_arc_length_m is None:
            # Compatibility for direct callers that predate metre-scale
            # trajectory progress.  Production callers pass the milestone arc.
            expert_target_arc_length_m = (
                self.start_arc_length_m + self.expert_progress_delta
            )
        expert_target_arc_length_m = float(expert_target_arc_length_m)
        if (
            not np.isfinite(expert_target_arc_length_m)
            or expert_target_arc_length_m < self.start_arc_length_m
        ):
            raise ValueError(
                "expert_target_arc_length_m must be finite and not precede "
                "the start arc length"
            )
        self.expert_target_arc_length_m = expert_target_arc_length_m
        self.expert_motion_m = max(
            0.0,
            self.expert_target_arc_length_m - self.start_arc_length_m,
        )
        self.grading_mode = (
            "moving"
            if self.expert_motion_m >= float(self.config.reference_motion_floor_m)
            else "stationary"
        )
        self.required_motion_m = (
            float(self.config.expert_progress_fraction) * self.expert_motion_m
            if self.grading_mode == "moving"
            else 0.0
        )
        self.required_progress_delta = (
            float(self.config.expert_progress_fraction) * self.expert_progress_delta
            if self.grading_mode == "moving"
            else 0.0
        )
        self.stationary_path_limit_m = min(
            float(self.config.max_path_deviation),
            max(
                float(self.config.stationary_path_tolerance_m),
                float(progress.translation_distance)
                + float(self.config.stationary_path_tolerance_m),
            ),
        )
        self.max_path_deviation = float(progress.translation_distance)
        self._best_progress = float(progress.progress)
        self._best_arc_length_m = float(progress.arc_length_m)
        self._steps_without_progress = 0
        self.active = bool(
            progress.translation_distance <= self.config.max_path_deviation
        )
        self.attempted_this_episode = self.active
        return self.active

    def should_force_rl(self, *, rl_available: bool) -> bool:
        return bool(self.active and self.remaining_steps > 0 and rl_available)

    def observe_rl_step(
        self,
        progress: TrajectoryProgress,
        *,
        success: bool = False,
        terminal: bool = False,
        intervention: bool = False,
        safety_abort: bool = False,
    ) -> Optional[RLProbeResult]:
        if not self.active:
            return None

        self.executed_steps += 1
        self.end_index = int(progress.index)
        self.end_progress = float(progress.progress)
        self.end_arc_length_m = float(progress.arc_length_m)
        self.max_path_deviation = max(
            self.max_path_deviation, float(progress.translation_distance)
        )
        if (
            self.grading_mode == "moving"
            and self.end_arc_length_m
            > self._best_arc_length_m + self.config.progress_epsilon_m
        ):
            self._best_progress = self.end_progress
            self._best_arc_length_m = self.end_arc_length_m
            self._steps_without_progress = 0
        elif self.grading_mode == "moving":
            self._steps_without_progress += 1
        else:
            # A stationary/micro-motion expert segment is graded by its tight
            # local corridor at the complete budget, not by synthetic progress.
            self._steps_without_progress = 0

        if intervention:
            return self.finish("intervention", safety_decrement=True)
        if safety_abort:
            return self.finish("safety_abort", safety_decrement=True)
        if progress.translation_distance > self.config.max_path_deviation:
            return self.finish("off_path", off_path_decrement=True)
        if success:
            return self.finish("success", success=True)
        if terminal:
            return self.finish("terminal_without_success")
        if (
            self.grading_mode == "moving"
            and self._steps_without_progress >= self.config.stall_steps
        ):
            return self.finish("stalled")
        if self.executed_steps >= self.budget_steps:
            return self.finish("budget_reached")
        return None

    def abort(
        self,
        reason: str,
        *,
        progress: Optional[TrajectoryProgress] = None,
        safety_decrement: bool = False,
    ) -> Optional[RLProbeResult]:
        return self.complete(
            reason,
            progress=progress,
            safety_decrement=safety_decrement,
        )

    def complete(
        self,
        reason: str,
        *,
        progress: Optional[TrajectoryProgress] = None,
        success: bool = False,
        off_path_decrement: bool = False,
        safety_decrement: bool = False,
    ) -> Optional[RLProbeResult]:
        if not self.active:
            return None
        if progress is not None:
            self.end_index = int(progress.index)
            self.end_progress = float(progress.progress)
            self.end_arc_length_m = float(progress.arc_length_m)
            self.max_path_deviation = max(
                self.max_path_deviation, float(progress.translation_distance)
            )
        return self.finish(
            reason,
            success=success,
            off_path_decrement=off_path_decrement,
            safety_decrement=safety_decrement,
        )

    def finish(
        self,
        reason: str,
        *,
        success: bool = False,
        off_path_decrement: bool = False,
        safety_decrement: bool = False,
    ) -> RLProbeResult:
        if not self.active:
            raise RuntimeError("Cannot finish an inactive RL probe")

        budget_before = int(self.budget_steps)
        progress_delta = max(0.0, self.end_progress - self.start_progress)
        actual_motion_m = max(
            0.0,
            self.end_arc_length_m - self.start_arc_length_m,
        )
        progress_ratio = (
            actual_motion_m / max(self.required_motion_m, np.finfo(np.float32).eps)
            if self.grading_mode == "moving"
            else 0.0
        )
        progress_qualified = bool(
            self.grading_mode == "moving" and actual_motion_m >= self.required_motion_m
        )
        stationary_qualified = bool(
            self.grading_mode == "stationary"
            and self.max_path_deviation <= self.stationary_path_limit_m
        )
        passed = bool(
            success
            or (
                reason == "budget_reached"
                and (progress_qualified or stationary_qualified)
                and self.max_path_deviation <= self.config.max_path_deviation
            )
        )
        if passed:
            self.pass_streak += 1
            if self.pass_streak >= self.config.required_passes:
                self.budget_steps = min(
                    self.config.max_steps,
                    self.budget_steps + self.config.step_increment,
                )
                self.pass_streak = 0
        else:
            self.pass_streak = 0
            if safety_decrement:
                self.budget_steps = max(
                    self.config.initial_steps,
                    self.budget_steps - self.config.safety_decrement,
                )
            elif off_path_decrement:
                self.budget_steps = max(
                    self.config.initial_steps,
                    self.budget_steps - self.config.off_path_decrement,
                )

        self.active = False
        return RLProbeResult(
            reason=str(reason),
            budget_before=budget_before,
            budget_after=int(self.budget_steps),
            executed_steps=int(self.executed_steps),
            start_index=int(self.start_index),
            end_index=int(self.end_index),
            expert_target_index=int(self.expert_target_index),
            start_progress=float(self.start_progress),
            end_progress=float(self.end_progress),
            expert_target_progress=float(self.expert_target_progress),
            expert_progress_delta=float(self.expert_progress_delta),
            required_progress_delta=float(self.required_progress_delta),
            progress_delta=float(progress_delta),
            progress_ratio=float(progress_ratio),
            start_arc_length_m=float(self.start_arc_length_m),
            end_arc_length_m=float(self.end_arc_length_m),
            expert_target_arc_length_m=float(self.expert_target_arc_length_m),
            expert_motion_m=float(self.expert_motion_m),
            required_motion_m=float(self.required_motion_m),
            actual_motion_m=float(actual_motion_m),
            grading_mode=str(self.grading_mode),
            stationary_path_limit_m=float(self.stationary_path_limit_m),
            max_path_deviation=float(self.max_path_deviation),
            success=bool(success),
            passed=passed,
            pass_streak=int(self.pass_streak),
        )
