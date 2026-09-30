"""Budgeted, region-local autonomous re-evaluation after policy changes.

The controller in this module is deliberately independent of robot and JAX
code.  It receives fixed-anchor policy signatures from the actor, tracks
evidence for each task region, and decides whether a non-RL Scheduler choice
should be replaced by one ordinary RL Option.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Mapping, Optional, Sequence

import numpy as np


POLICY_CHANGE_PROBE_STATE_SCHEMA_VERSION = "policy_change_probe_state_v2"
POLICY_CHANGE_PROBE_SCHEDULER_STATE_SCHEMA_VERSION = (
    "hilserl_option_history_policy_change_probe_v1"
)
POLICY_CHANGE_PROBE_OUTCOME_FEATURE_DIM = 8


def _finite_array(value: Any, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != int(ndim) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-D array, got {array.shape}")
    return array


@dataclass(frozen=True)
class PolicySignature:
    """Policy outputs on one region's fixed anchor observations.

    ``continuous_mean`` and ``continuous_std`` describe only continuous action
    dimensions that can affect execution.  For hybrid SAC, ``discrete_logits``
    holds the gripper Q outputs.  They are converted to a categorical diagnostic
    distribution with a fixed temperature when drift is evaluated.
    """

    continuous_mean: np.ndarray
    continuous_std: np.ndarray
    discrete_logits: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        mean = _finite_array(self.continuous_mean, name="continuous_mean", ndim=2)
        std = _finite_array(self.continuous_std, name="continuous_std", ndim=2)
        if mean.shape != std.shape or mean.shape[0] == 0 or mean.shape[1] == 0:
            raise ValueError(
                "continuous_mean/std must share a non-empty (anchors, actions) "
                f"shape, got {mean.shape} and {std.shape}"
            )
        if np.any(std <= 0.0):
            raise ValueError("continuous_std must be strictly positive")
        logits = self.discrete_logits
        if logits is not None:
            logits = _finite_array(logits, name="discrete_logits", ndim=2)
            if logits.shape[0] != mean.shape[0] or logits.shape[1] < 2:
                raise ValueError(
                    "discrete_logits must have the same anchor count and at least "
                    f"two categories, got {logits.shape}"
                )
        object.__setattr__(self, "continuous_mean", mean.copy())
        object.__setattr__(self, "continuous_std", std.copy())
        object.__setattr__(
            self,
            "discrete_logits",
            None if logits is None else logits.copy(),
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "continuous_mean": self.continuous_mean.copy(),
            "continuous_std": self.continuous_std.copy(),
            "discrete_logits": (
                None if self.discrete_logits is None else self.discrete_logits.copy()
            ),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "PolicySignature":
        return cls(
            continuous_mean=state["continuous_mean"],
            continuous_std=state["continuous_std"],
            discrete_logits=state.get("discrete_logits"),
        )


@dataclass(frozen=True)
class PolicyDrift:
    continuous_skl: float
    discrete_skl: float
    combined_skl: float


def _categorical_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= np.max(scaled, axis=-1, keepdims=True)
    probabilities = np.exp(scaled)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return np.clip(probabilities, 1e-12, 1.0)


def policy_signature_symmetric_kl(
    current: PolicySignature,
    reference: PolicySignature,
    *,
    discrete_temperature: float = 1.0,
    discrete_weight: float = 1.0,
) -> PolicyDrift:
    """Return anchor- and action-normalized symmetric KL policy drift."""

    if current.continuous_mean.shape != reference.continuous_mean.shape:
        raise ValueError(
            "Current/reference continuous signature shapes differ: "
            f"{current.continuous_mean.shape} vs {reference.continuous_mean.shape}"
        )
    if not np.isfinite(discrete_temperature) or float(discrete_temperature) <= 0.0:
        raise ValueError("discrete_temperature must be finite and positive")
    if not np.isfinite(discrete_weight) or float(discrete_weight) < 0.0:
        raise ValueError("discrete_weight must be finite and non-negative")

    current_mean = current.continuous_mean.astype(np.float64)
    reference_mean = reference.continuous_mean.astype(np.float64)
    current_var = np.square(current.continuous_std.astype(np.float64))
    reference_var = np.square(reference.continuous_std.astype(np.float64))
    mean_delta_sq = np.square(current_mean - reference_mean)

    # Averaging KL(p||q) and KL(q||p) cancels the log-scale terms.
    gaussian_skl_per_dimension = 0.25 * (
        (current_var + mean_delta_sq) / reference_var
        + (reference_var + mean_delta_sq) / current_var
        - 2.0
    )
    continuous_skl = float(np.mean(gaussian_skl_per_dimension))

    current_logits = current.discrete_logits
    reference_logits = reference.discrete_logits
    if (current_logits is None) != (reference_logits is None):
        raise ValueError("Current/reference discrete signature presence differs")
    discrete_skl = 0.0
    discrete_components = 0.0
    if current_logits is not None:
        if current_logits.shape != reference_logits.shape:
            raise ValueError(
                "Current/reference discrete signature shapes differ: "
                f"{current_logits.shape} vs {reference_logits.shape}"
            )
        current_prob = _categorical_probabilities(current_logits, discrete_temperature)
        reference_prob = _categorical_probabilities(
            reference_logits, discrete_temperature
        )
        kl_current_reference = np.sum(
            current_prob * (np.log(current_prob) - np.log(reference_prob)), axis=-1
        )
        kl_reference_current = np.sum(
            reference_prob * (np.log(reference_prob) - np.log(current_prob)),
            axis=-1,
        )
        discrete_skl = float(
            np.mean(0.5 * (kl_current_reference + kl_reference_current))
        )
        discrete_components = float(discrete_weight)

    continuous_components = float(current.continuous_mean.shape[1])
    combined_skl = (
        continuous_skl * continuous_components + discrete_skl * discrete_components
    ) / (continuous_components + discrete_components)
    return PolicyDrift(
        continuous_skl=max(0.0, continuous_skl),
        discrete_skl=max(0.0, discrete_skl),
        combined_skl=max(0.0, float(combined_skl)),
    )


@dataclass(frozen=True)
class RegionalAnchorSet:
    """Fixed observations and demonstration-index regions for one task."""

    region_names: tuple[str, ...]
    region_start_indices: tuple[int, ...]
    anchor_observations: tuple[tuple[Mapping[str, Any], ...], ...]
    trajectory_length: int
    fingerprint: str

    def __post_init__(self) -> None:
        region_count = len(self.region_names)
        if region_count == 0 or len(set(self.region_names)) != region_count:
            raise ValueError("region_names must be non-empty and unique")
        if len(self.region_start_indices) != region_count:
            raise ValueError("region_start_indices must match region_names")
        if len(self.anchor_observations) != region_count:
            raise ValueError("anchor_observations must match region_names")
        if int(self.trajectory_length) <= 0:
            raise ValueError("trajectory_length must be positive")
        starts = tuple(int(index) for index in self.region_start_indices)
        if starts[0] != 0 or any(
            left >= right for left, right in zip(starts, starts[1:])
        ):
            raise ValueError(
                "region_start_indices must start at zero and increase strictly"
            )
        if starts[-1] >= int(self.trajectory_length):
            raise ValueError("last region start is outside the trajectory")
        if any(len(region_anchors) == 0 for region_anchors in self.anchor_observations):
            raise ValueError("every region must contain at least one anchor")
        if not str(self.fingerprint):
            raise ValueError("fingerprint must be non-empty")

    @property
    def region_count(self) -> int:
        return len(self.region_names)

    def region_for_progress_index(self, progress_index: int) -> int:
        index = int(np.clip(int(progress_index), 0, self.trajectory_length - 1))
        return int(np.searchsorted(self.region_start_indices, index, side="right") - 1)

    @property
    def region_stop_indices(self) -> tuple[int, ...]:
        return (*self.region_start_indices[1:], int(self.trajectory_length))

    def region_transition_counts(self) -> tuple[int, ...]:
        """Return demo transitions assigned to each region.

        The anchor source is a transition sequence (one item per environment
        step), so ``stop - start`` is already a primitive-step reference.
        """

        return tuple(
            int(stop) - int(start)
            for start, stop in zip(self.region_start_indices, self.region_stop_indices)
        )

    def region_max_horizons(self, step_quantum: int) -> tuple[int, ...]:
        """Round each demo-derived regional length up to an Option quantum."""

        step_quantum = int(step_quantum)
        if step_quantum <= 0:
            raise ValueError("step_quantum must be positive")
        return tuple(
            max(
                step_quantum,
                ((count + step_quantum - 1) // step_quantum) * step_quantum,
            )
            for count in self.region_transition_counts()
        )

    def remaining_region_steps(self, progress_index: int) -> int:
        index = int(np.clip(int(progress_index), 0, self.trajectory_length - 1))
        region = self.region_for_progress_index(index)
        return max(1, int(self.region_stop_indices[region]) - index)


@dataclass(frozen=True)
class PolicyChangeProbeConfig:
    region_names: tuple[str, ...]
    drift_threshold: float
    max_age_steps: int
    budget_window_steps: int
    budget_steps: int
    rl_option_horizon: int
    region_max_horizons: tuple[int, ...]
    initial_horizon_steps: int = 5
    horizon_increment_steps: int = 5
    required_passes: int = 2
    off_path_decrement_steps: int = 5
    safety_decrement_steps: int = 10
    max_path_deviation_m: float = 0.08
    evidence_history_length: int = 4
    task_return_scale: float = 1.0
    discrete_temperature: float = 1.0
    discrete_weight: float = 1.0
    use_policy_drift: bool = True

    def __post_init__(self) -> None:
        if not self.region_names or len(set(self.region_names)) != len(
            self.region_names
        ):
            raise ValueError("region_names must be non-empty and unique")
        if not np.isfinite(self.drift_threshold) or self.drift_threshold <= 0.0:
            raise ValueError("drift_threshold must be finite and positive")
        if int(self.max_age_steps) <= 0:
            raise ValueError("max_age_steps must be positive")
        if int(self.budget_window_steps) <= 0:
            raise ValueError("budget_window_steps must be positive")
        if not 0 <= int(self.budget_steps) <= int(self.budget_window_steps):
            raise ValueError("budget_steps must be in [0, budget_window_steps]")
        if int(self.rl_option_horizon) <= 0:
            raise ValueError("rl_option_horizon must be positive")
        if int(self.rl_option_horizon) > int(self.budget_window_steps):
            raise ValueError("rl_option_horizon cannot exceed budget_window_steps")
        if 0 < int(self.budget_steps) < int(self.rl_option_horizon):
            raise ValueError("a non-zero budget_steps must fund at least one RL Option")
        if len(self.region_max_horizons) != len(self.region_names):
            raise ValueError("region_max_horizons must match region_names")
        for horizon in self.region_max_horizons:
            if (
                int(horizon) < int(self.rl_option_horizon)
                or int(horizon) % int(self.rl_option_horizon) != 0
            ):
                raise ValueError(
                    "region_max_horizons must be positive multiples of "
                    "rl_option_horizon"
                )
        if (
            int(self.initial_horizon_steps) < int(self.rl_option_horizon)
            or int(self.initial_horizon_steps) % int(self.rl_option_horizon) != 0
        ):
            raise ValueError(
                "initial_horizon_steps must be a positive multiple of "
                "rl_option_horizon"
            )
        if (
            int(self.horizon_increment_steps) <= 0
            or int(self.horizon_increment_steps) % int(self.rl_option_horizon) != 0
        ):
            raise ValueError(
                "horizon_increment_steps must be a positive multiple of "
                "rl_option_horizon"
            )
        if int(self.required_passes) <= 0:
            raise ValueError("required_passes must be positive")
        if int(self.off_path_decrement_steps) < 0:
            raise ValueError("off_path_decrement_steps must be non-negative")
        if int(self.safety_decrement_steps) < 0:
            raise ValueError("safety_decrement_steps must be non-negative")
        if (
            not np.isfinite(self.max_path_deviation_m)
            or float(self.max_path_deviation_m) <= 0.0
        ):
            raise ValueError("max_path_deviation_m must be finite and positive")
        if int(self.evidence_history_length) <= 0:
            raise ValueError("evidence_history_length must be positive")
        if not np.isfinite(self.task_return_scale) or self.task_return_scale <= 0.0:
            raise ValueError("task_return_scale must be finite and positive")
        if (
            not np.isfinite(self.discrete_temperature)
            or self.discrete_temperature <= 0.0
        ):
            raise ValueError("discrete_temperature must be finite and positive")
        if not np.isfinite(self.discrete_weight) or self.discrete_weight < 0.0:
            raise ValueError("discrete_weight must be finite and non-negative")

    @property
    def state_feature_dim(self) -> int:
        # region one-hot + seen/age/combined/continuous/discrete/budget + records
        return (
            len(self.region_names)
            + 6
            + int(self.evidence_history_length)
            * POLICY_CHANGE_PROBE_OUTCOME_FEATURE_DIM
        )

    def identity_dict(self) -> Mapping[str, Any]:
        return {
            "use_policy_drift": bool(self.use_policy_drift),
            "region_names": tuple(self.region_names),
            "drift_threshold": float(self.drift_threshold),
            "max_age_steps": int(self.max_age_steps),
            "budget_window_steps": int(self.budget_window_steps),
            "budget_steps": int(self.budget_steps),
            "rl_option_horizon": int(self.rl_option_horizon),
            "region_max_horizons": tuple(
                int(value) for value in self.region_max_horizons
            ),
            "initial_horizon_steps": int(self.initial_horizon_steps),
            "horizon_increment_steps": int(self.horizon_increment_steps),
            "required_passes": int(self.required_passes),
            "off_path_decrement_steps": int(self.off_path_decrement_steps),
            "safety_decrement_steps": int(self.safety_decrement_steps),
            "max_path_deviation_m": float(self.max_path_deviation_m),
            "evidence_history_length": int(self.evidence_history_length),
            "task_return_scale": float(self.task_return_scale),
            "discrete_temperature": float(self.discrete_temperature),
            "discrete_weight": float(self.discrete_weight),
        }


@dataclass(frozen=True)
class AutonomousOutcome:
    end_step: int
    task_return: float
    duration: int
    termination_reason: str
    success: bool
    episode_terminated: bool
    episode_truncated: bool
    was_probe: bool

    def __post_init__(self) -> None:
        if int(self.end_step) < 0:
            raise ValueError("end_step must be non-negative")
        if int(self.duration) <= 0:
            raise ValueError("duration must be positive")
        if not np.isfinite(self.task_return):
            raise ValueError("task_return must be finite")
        if not str(self.termination_reason):
            raise ValueError("termination_reason must be non-empty")

    def features(self, config: PolicyChangeProbeConfig) -> np.ndarray:
        safety = self.termination_reason in {
            "human_intervention",
            "safety_interruption",
            "execution_error",
        }
        timeout = self.termination_reason == "horizon_reached"
        return np.asarray(
            [
                1.0,
                np.clip(
                    float(self.task_return) / float(config.task_return_scale),
                    -1.0,
                    1.0,
                ),
                np.clip(
                    float(self.duration) / float(config.rl_option_horizon),
                    0.0,
                    1.0,
                ),
                float(self.success),
                float(self.episode_terminated),
                float(self.episode_truncated),
                float(timeout),
                float(safety),
            ],
            dtype=np.float32,
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "end_step": int(self.end_step),
            "task_return": float(self.task_return),
            "duration": int(self.duration),
            "termination_reason": str(self.termination_reason),
            "success": bool(self.success),
            "episode_terminated": bool(self.episode_terminated),
            "episode_truncated": bool(self.episode_truncated),
            "was_probe": bool(self.was_probe),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "AutonomousOutcome":
        return cls(**dict(state))


@dataclass(frozen=True)
class PolicyChangeProbeDecision:
    override: bool
    requested: bool
    reason: str
    trigger_reasons: tuple[str, ...]
    region: int
    region_name: str
    age_steps: Optional[int]
    drift: Optional[PolicyDrift]
    remaining_budget_steps: int
    budget_window_index: int
    session_horizon_steps: int
    region_max_horizon_steps: int
    session_continuation: bool = False

    def metrics(self) -> Mapping[str, float]:
        drift = self.drift
        values = {
            "policy_change_probe/requested": float(self.requested),
            "policy_change_probe/override": float(self.override),
            "policy_change_probe/age_steps": float(
                -1 if self.age_steps is None else self.age_steps
            ),
            "policy_change_probe/drift": float(
                0.0 if drift is None else drift.combined_skl
            ),
            "policy_change_probe/continuous_drift": float(
                0.0 if drift is None else drift.continuous_skl
            ),
            "policy_change_probe/discrete_drift": float(
                0.0 if drift is None else drift.discrete_skl
            ),
            "policy_change_probe/remaining_budget_steps": float(
                self.remaining_budget_steps
            ),
            "policy_change_probe/session_horizon_steps": float(
                self.session_horizon_steps
            ),
            "policy_change_probe/region_max_horizon_steps": float(
                self.region_max_horizon_steps
            ),
            "policy_change_probe/session_continuation": float(
                self.session_continuation
            ),
            f"policy_change_probe/reason/{self.reason}": 1.0,
            f"policy_change_probe/region/{self.region_name}": 1.0,
        }
        for trigger_reason in self.trigger_reasons:
            values[f"policy_change_probe/trigger/{trigger_reason}"] = 1.0
        return values


class _RegionEvidence:
    def __init__(self, history_length: int) -> None:
        self.last_autonomous_step: Optional[int] = None
        self.reference_signature: Optional[PolicySignature] = None
        self.outcomes: Deque[AutonomousOutcome] = deque(maxlen=int(history_length))
        self.current_horizon_steps = 0
        self.pass_streak = 0


@dataclass
class _ProbeSession:
    region: int
    start_step: int
    budget_window_index: int
    target_horizon_steps: int
    completed_steps: int
    task_return: float
    max_path_deviation_m: float
    signature: Optional[PolicySignature]
    trigger_reasons: tuple[str, ...]


class PolicyChangeProbeController:
    """Track local evidence and gate budgeted autonomous re-evaluation."""

    def __init__(
        self,
        config: PolicyChangeProbeConfig,
        *,
        anchor_fingerprint: str,
    ) -> None:
        if not str(anchor_fingerprint):
            raise ValueError("anchor_fingerprint must be non-empty")
        self.config = config
        self.anchor_fingerprint = str(anchor_fingerprint)
        self._regions = [
            _RegionEvidence(config.evidence_history_length) for _ in config.region_names
        ]
        for region, evidence in enumerate(self._regions):
            evidence.current_horizon_steps = min(
                int(config.initial_horizon_steps),
                int(config.region_max_horizons[region]),
            )
        self._current_signatures: Optional[tuple[PolicySignature, ...]] = None
        self.current_policy_version: Optional[int] = None
        self._budget_window_index: Optional[int] = None
        self._budget_spent_steps = 0
        self._active_reservation: Optional[tuple[int, int, int, int]] = None
        self._active_session: Optional[_ProbeSession] = None

    @property
    def state_feature_dim(self) -> int:
        return self.config.state_feature_dim

    def _validate_region(self, region: int) -> int:
        region = int(region)
        if not 0 <= region < len(self._regions):
            raise IndexError(f"region {region} is outside configured regions")
        return region

    def set_current_signatures(
        self,
        signatures: Sequence[PolicySignature],
        *,
        policy_version: int,
    ) -> None:
        if not self.config.use_policy_drift:
            self.current_policy_version = int(policy_version)
            return
        signatures = tuple(signatures)
        if len(signatures) != len(self._regions):
            raise ValueError("signatures must contain exactly one item per region")
        for region, signature in enumerate(signatures):
            if not isinstance(signature, PolicySignature):
                raise TypeError("signatures must contain PolicySignature values")
            reference = self._regions[region].reference_signature
            if reference is not None:
                policy_signature_symmetric_kl(
                    signature,
                    reference,
                    discrete_temperature=self.config.discrete_temperature,
                    discrete_weight=self.config.discrete_weight,
                )
        self._current_signatures = signatures
        self.current_policy_version = int(policy_version)

    def current_signature(self, region: int) -> Optional[PolicySignature]:
        region = self._validate_region(region)
        if not self.config.use_policy_drift:
            return None
        if self._current_signatures is None:
            raise RuntimeError("current policy signatures have not been set")
        return self._current_signatures[region]

    def drift(self, region: int) -> Optional[PolicyDrift]:
        region = self._validate_region(region)
        if not self.config.use_policy_drift:
            return None
        reference = self._regions[region].reference_signature
        if reference is None:
            return None
        return policy_signature_symmetric_kl(
            self.current_signature(region),
            reference,
            discrete_temperature=self.config.discrete_temperature,
            discrete_weight=self.config.discrete_weight,
        )

    def _budget_snapshot(self, step: int) -> tuple[int, int, int]:
        step = int(step)
        if step < 0:
            raise ValueError("step must be non-negative")
        window_index = step // int(self.config.budget_window_steps)
        if self._budget_window_index != window_index:
            self._budget_window_index = window_index
            self._budget_spent_steps = 0
        remaining = max(
            0, int(self.config.budget_steps) - int(self._budget_spent_steps)
        )
        steps_until_window_end = (window_index + 1) * int(
            self.config.budget_window_steps
        ) - step
        return window_index, remaining, steps_until_window_end

    def decide(
        self,
        *,
        base_is_rl: bool,
        rl_available: bool,
        region: int,
        step: int,
        remaining_region_steps: Optional[int] = None,
    ) -> PolicyChangeProbeDecision:
        region = self._validate_region(region)
        evidence = self._regions[region]
        window_index, remaining, steps_until_window_end = self._budget_snapshot(step)
        age_steps = (
            None
            if evidence.last_autonomous_step is None
            else max(0, int(step) - int(evidence.last_autonomous_step))
        )
        drift = self.drift(region)
        session_continuation = self._active_session is not None
        if session_continuation and self._active_session.region != region:
            raise ValueError(
                "active policy-change Probe session crossed a region boundary "
                "without being completed"
            )
        trigger_reasons = []
        if session_continuation:
            trigger_reasons.append("session_continue")
        elif evidence.last_autonomous_step is None:
            trigger_reasons.append("unseen")
        else:
            if drift is not None and drift.combined_skl >= self.config.drift_threshold:
                trigger_reasons.append("policy_change")
            if age_steps is not None and age_steps >= self.config.max_age_steps:
                trigger_reasons.append("age")
        requested = bool(trigger_reasons)

        if base_is_rl and not session_continuation:
            reason = "base_rl"
            override = False
        elif not requested:
            reason = "fresh_evidence"
            override = False
        elif not rl_available:
            reason = "rl_unavailable"
            override = False
        elif remaining < self.config.rl_option_horizon:
            reason = "budget_exhausted"
            override = False
        elif steps_until_window_end < self.config.rl_option_horizon:
            reason = "window_boundary"
            override = False
        else:
            reason = trigger_reasons[0]
            override = True

        if session_continuation:
            session_horizon = int(self._active_session.target_horizon_steps)
        else:
            session_horizon = int(evidence.current_horizon_steps)
            if remaining_region_steps is not None:
                remaining_region_steps = max(1, int(remaining_region_steps))
                quantum = int(self.config.rl_option_horizon)
                rounded_remaining = (
                    (remaining_region_steps + quantum - 1) // quantum
                ) * quantum
                session_horizon = min(session_horizon, rounded_remaining)

        return PolicyChangeProbeDecision(
            override=override,
            requested=requested,
            reason=reason,
            trigger_reasons=tuple(trigger_reasons),
            region=region,
            region_name=self.config.region_names[region],
            age_steps=age_steps,
            drift=drift,
            remaining_budget_steps=remaining,
            budget_window_index=window_index,
            session_horizon_steps=session_horizon,
            region_max_horizon_steps=int(self.config.region_max_horizons[region]),
            session_continuation=session_continuation,
        )

    def reserve_probe(
        self,
        decision: PolicyChangeProbeDecision,
        *,
        start_step: int,
    ) -> None:
        """Reserve one full RL horizon before any probe robot interaction.

        Completion refunds unused steps, so normal accounting uses actual
        duration. If the actor exits mid-Option, the persisted reservation is
        deliberately retained as a conservative full-horizon charge.
        """

        if not decision.override:
            raise ValueError("cannot reserve a denied policy-change Probe")
        if self._active_reservation is not None:
            raise RuntimeError("a policy-change Probe reservation is already active")
        start_step = int(start_step)
        window_index, remaining, steps_until_window_end = self._budget_snapshot(
            start_step
        )
        horizon = int(self.config.rl_option_horizon)
        if window_index != decision.budget_window_index:
            raise ValueError("policy-change Probe decision belongs to another window")
        if remaining < horizon or steps_until_window_end < horizon:
            raise ValueError("policy-change Probe budget is no longer available")
        self._budget_spent_steps += horizon
        self._active_reservation = (
            int(window_index),
            int(decision.region),
            start_step,
            horizon,
        )
        if self._active_session is None:
            self._active_session = _ProbeSession(
                region=int(decision.region),
                start_step=start_step,
                budget_window_index=int(decision.budget_window_index),
                target_horizon_steps=int(decision.session_horizon_steps),
                completed_steps=0,
                task_return=0.0,
                max_path_deviation_m=0.0,
                signature=PolicySignature.from_state_dict(
                    self.current_signature(decision.region).state_dict()
                )
                if self.config.use_policy_drift
                else None,
                trigger_reasons=tuple(decision.trigger_reasons),
            )
        elif (
            self._active_session.region != int(decision.region)
            or not decision.session_continuation
        ):
            raise ValueError("probe continuation does not match the active session")

    def cancel_probe_reservation(
        self,
        decision: PolicyChangeProbeDecision,
        *,
        start_step: int,
    ) -> None:
        """Release a reservation when no primitive interaction occurred."""

        start_step = int(start_step)
        expected_reservation = (
            int(decision.budget_window_index),
            int(decision.region),
            start_step,
            int(self.config.rl_option_horizon),
        )
        if self._active_reservation != expected_reservation:
            raise ValueError("probe cancellation has no matching active reservation")
        self._budget_spent_steps -= int(self.config.rl_option_horizon)
        if self._budget_spent_steps < 0:
            raise RuntimeError("policy-change Probe budget accounting underflow")
        self._active_reservation = None
        self._active_session = None

    @property
    def probe_session_active(self) -> bool:
        return self._active_session is not None

    def current_horizon_steps(self, region: int) -> int:
        return int(self._regions[self._validate_region(region)].current_horizon_steps)

    def pass_streak(self, region: int) -> int:
        return int(self._regions[self._validate_region(region)].pass_streak)

    def observe_probe_path(self, path_deviation_m: float) -> None:
        """Accumulate the maximum actual TCP deviation during a Probe session."""

        if self._active_session is None:
            return
        path_deviation_m = float(path_deviation_m)
        if not np.isfinite(path_deviation_m) or path_deviation_m < 0.0:
            raise ValueError("path_deviation_m must be finite and non-negative")
        self._active_session.max_path_deviation_m = max(
            float(self._active_session.max_path_deviation_m), path_deviation_m
        )

    def abort_probe_session(
        self, *, end_step: int, termination_reason: str
    ) -> Optional[AutonomousOutcome]:
        """Fail a between-chunk session when Scheduler control is interrupted."""

        if self._active_reservation is not None:
            raise RuntimeError(
                "cannot abort a Probe session with an active reservation"
            )
        session = self._active_session
        if session is None:
            return None
        if session.completed_steps <= 0:
            self._active_session = None
            return None
        evidence = self._regions[session.region]
        evidence.pass_streak = 0
        evidence.current_horizon_steps = max(
            int(self.config.rl_option_horizon),
            int(evidence.current_horizon_steps)
            - int(self.config.safety_decrement_steps),
        )
        outcome = AutonomousOutcome(
            end_step=max(
                int(end_step), int(session.start_step + session.completed_steps)
            ),
            task_return=float(session.task_return),
            duration=int(session.completed_steps),
            termination_reason=str(termination_reason),
            success=False,
            episode_terminated=False,
            episode_truncated=True,
            was_probe=True,
        )
        evidence.last_autonomous_step = int(outcome.end_step)
        evidence.reference_signature = session.signature
        evidence.outcomes.append(outcome)
        self._active_session = None
        return outcome

    def record_autonomous_execution(
        self,
        *,
        region: int,
        signature: Optional[PolicySignature],
        start_step: int,
        duration: int,
        task_return: float,
        termination_reason: str,
        success: bool,
        episode_terminated: bool,
        episode_truncated: bool,
        probe_decision: Optional[PolicyChangeProbeDecision] = None,
        final_region: Optional[int] = None,
        path_deviation_m: Optional[float] = None,
    ) -> AutonomousOutcome:
        region = self._validate_region(region)
        start_step = int(start_step)
        duration = int(duration)
        task_return = float(task_return)
        if start_step < 0 or duration <= 0:
            raise ValueError("start_step must be non-negative and duration positive")
        if duration > self.config.rl_option_horizon:
            raise ValueError("autonomous duration exceeds configured RL Option horizon")
        if not np.isfinite(task_return):
            raise ValueError("task_return must be finite")
        # Validate signature structure against the current anchor set.
        if self.config.use_policy_drift:
            policy_signature_symmetric_kl(
                self.current_signature(region),
                signature,
                discrete_temperature=self.config.discrete_temperature,
                discrete_weight=self.config.discrete_weight,
            )

        was_probe = probe_decision is not None
        if was_probe:
            if not probe_decision.override or probe_decision.region != region:
                raise ValueError("probe_decision does not authorize this execution")
            start_window = start_step // int(self.config.budget_window_steps)
            end_window = (start_step + duration - 1) // int(
                self.config.budget_window_steps
            )
            if (
                start_window != probe_decision.budget_window_index
                or end_window != start_window
            ):
                raise ValueError("probe execution crossed its authorized budget window")
            reservation = self._active_reservation
            expected_reservation = (
                int(start_window),
                region,
                start_step,
                int(self.config.rl_option_horizon),
            )
            if reservation != expected_reservation:
                raise ValueError("probe execution has no matching active reservation")
            self._budget_spent_steps -= int(self.config.rl_option_horizon) - duration
            self._active_reservation = None

        end_step = start_step + duration
        outcome = AutonomousOutcome(
            end_step=end_step,
            task_return=task_return,
            duration=duration,
            termination_reason=str(termination_reason),
            success=bool(success),
            episode_terminated=bool(episode_terminated),
            episode_truncated=bool(episode_truncated),
            was_probe=was_probe,
        )
        evidence = self._regions[region]
        if was_probe:
            session = self._active_session
            if session is None or session.region != region:
                raise ValueError("probe execution has no matching active session")
            session.completed_steps += duration
            session.task_return += task_return
            if path_deviation_m is not None and (
                not np.isfinite(path_deviation_m) or float(path_deviation_m) < 0.0
            ):
                raise ValueError("path_deviation_m must be finite and non-negative")
            if path_deviation_m is not None:
                self.observe_probe_path(path_deviation_m)
            final_region = (
                region if final_region is None else self._validate_region(final_region)
            )
            safety = str(termination_reason) in {
                "human_intervention",
                "safety_interruption",
                "execution_error",
            }
            off_path = bool(
                final_region < region
                or (
                    session.max_path_deviation_m
                    > float(self.config.max_path_deviation_m)
                )
            )
            region_completed = final_region > region
            target_completed = session.completed_steps >= session.target_horizon_steps
            interrupted = str(termination_reason) != "horizon_reached"
            end_window, remaining, steps_until_window_end = self._budget_snapshot(
                end_step
            )
            budget_blocked = bool(
                not target_completed
                and (
                    end_window != session.budget_window_index
                    or remaining < int(self.config.rl_option_horizon)
                    or steps_until_window_end < int(self.config.rl_option_horizon)
                )
            )
            session_complete = bool(
                success
                or safety
                or off_path
                or region_completed
                or target_completed
                or interrupted
                or budget_blocked
            )
            if not session_complete:
                return outcome

            passed = bool(
                not safety
                and not off_path
                and (
                    success
                    or region_completed
                    or (target_completed and not interrupted)
                )
            )
            horizon_before = int(evidence.current_horizon_steps)
            if passed:
                evidence.pass_streak += 1
                if evidence.pass_streak >= int(self.config.required_passes):
                    evidence.current_horizon_steps = min(
                        int(self.config.region_max_horizons[region]),
                        horizon_before + int(self.config.horizon_increment_steps),
                    )
                    evidence.pass_streak = 0
            else:
                evidence.pass_streak = 0
                decrement = (
                    int(self.config.safety_decrement_steps)
                    if safety
                    else int(self.config.off_path_decrement_steps)
                )
                if not budget_blocked:
                    evidence.current_horizon_steps = max(
                        int(self.config.rl_option_horizon),
                        horizon_before - decrement,
                    )
            completion_reason = (
                "safety"
                if safety
                else "off_path"
                if off_path
                else "success"
                if success
                else "region_boundary"
                if region_completed
                else "target_horizon"
                if target_completed
                else "budget_exhausted"
                if budget_blocked
                else str(termination_reason)
            )
            outcome = AutonomousOutcome(
                end_step=end_step,
                task_return=float(session.task_return),
                duration=int(session.completed_steps),
                termination_reason=completion_reason,
                success=passed,
                episode_terminated=bool(episode_terminated),
                episode_truncated=bool(episode_truncated),
                was_probe=True,
            )
            signature = session.signature
            self._active_session = None
        evidence.last_autonomous_step = end_step
        evidence.reference_signature = (
            PolicySignature.from_state_dict(signature.state_dict())
            if self.config.use_policy_drift
            else None
        )
        evidence.outcomes.append(outcome)
        return outcome

    def state_features(self, *, region: int, step: int) -> np.ndarray:
        region = self._validate_region(region)
        evidence = self._regions[region]
        _, remaining, _ = self._budget_snapshot(step)
        region_one_hot = np.zeros(len(self._regions), dtype=np.float32)
        region_one_hot[region] = 1.0
        seen = evidence.last_autonomous_step is not None
        age_steps = (
            self.config.max_age_steps
            if not seen
            else max(0, int(step) - int(evidence.last_autonomous_step))
        )
        drift = self.drift(region)
        combined = 0.0 if drift is None else drift.combined_skl
        continuous = 0.0 if drift is None else drift.continuous_skl
        discrete = 0.0 if drift is None else drift.discrete_skl
        scale = float(self.config.drift_threshold)
        scalars = np.asarray(
            [
                float(seen),
                np.clip(float(age_steps) / float(self.config.max_age_steps), 0.0, 1.0),
                np.clip(combined / scale, 0.0, 1.0),
                np.clip(continuous / scale, 0.0, 1.0),
                np.clip(discrete / scale, 0.0, 1.0),
                (
                    0.0
                    if self.config.budget_steps == 0
                    else float(remaining) / float(self.config.budget_steps)
                ),
            ],
            dtype=np.float32,
        )
        history = np.zeros(
            (
                self.config.evidence_history_length,
                POLICY_CHANGE_PROBE_OUTCOME_FEATURE_DIM,
            ),
            dtype=np.float32,
        )
        if evidence.outcomes:
            outcome_features = np.stack(
                [outcome.features(self.config) for outcome in evidence.outcomes]
            )
            history[-len(outcome_features) :] = outcome_features
        features = np.concatenate((region_one_hot, scalars, history.reshape(-1)))
        if features.shape != (self.state_feature_dim,) or not np.all(
            np.isfinite(features)
        ):
            raise ValueError(f"invalid policy-change evidence state {features.shape}")
        return features.astype(np.float32, copy=False)

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": POLICY_CHANGE_PROBE_STATE_SCHEMA_VERSION,
            "config": dict(self.config.identity_dict()),
            "anchor_fingerprint": self.anchor_fingerprint,
            "budget_window_index": self._budget_window_index,
            "budget_spent_steps": int(self._budget_spent_steps),
            "regions": [
                {
                    "last_autonomous_step": evidence.last_autonomous_step,
                    "reference_signature": (
                        None
                        if evidence.reference_signature is None
                        else evidence.reference_signature.state_dict()
                    ),
                    "outcomes": [outcome.state_dict() for outcome in evidence.outcomes],
                    "current_horizon_steps": int(evidence.current_horizon_steps),
                    "pass_streak": int(evidence.pass_streak),
                }
                for evidence in self._regions
            ],
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != POLICY_CHANGE_PROBE_STATE_SCHEMA_VERSION:
            raise ValueError("incompatible policy-change Probe state schema")
        saved_config = dict(state.get("config", {}))
        # Legacy snapshots have no mode field. Preserve their counters and
        # outcomes when switching off drift; never reuse their signatures.
        saved_config.setdefault("use_policy_drift", True)
        if not self.config.use_policy_drift:
            saved_config["use_policy_drift"] = False
        if saved_config != dict(self.config.identity_dict()):
            raise ValueError("policy-change Probe config differs from saved state")
        if state.get("anchor_fingerprint") != self.anchor_fingerprint:
            raise ValueError("policy-change Probe anchor fingerprint differs")
        region_states = state.get("regions")
        if not isinstance(region_states, Sequence) or len(region_states) != len(
            self._regions
        ):
            raise ValueError("saved policy-change Probe region count differs")

        restored_regions = []
        for region, region_state in enumerate(region_states):
            evidence = _RegionEvidence(self.config.evidence_history_length)
            evidence.current_horizon_steps = int(
                region_state.get(
                    "current_horizon_steps",
                    min(
                        int(self.config.initial_horizon_steps),
                        int(self.config.region_max_horizons[region]),
                    ),
                )
            )
            if not (
                int(self.config.rl_option_horizon)
                <= evidence.current_horizon_steps
                <= int(self.config.region_max_horizons[region])
            ):
                raise ValueError("saved regional Probe horizon is invalid")
            evidence.pass_streak = int(region_state.get("pass_streak", 0))
            if not 0 <= evidence.pass_streak < int(self.config.required_passes):
                raise ValueError("saved regional Probe pass streak is invalid")
            last_step = region_state.get("last_autonomous_step")
            evidence.last_autonomous_step = (
                None if last_step is None else int(last_step)
            )
            signature_state = region_state.get("reference_signature")
            evidence.reference_signature = (
                None
                if signature_state is None or not self.config.use_policy_drift
                else PolicySignature.from_state_dict(signature_state)
            )
            outcomes = region_state.get("outcomes", ())
            if len(outcomes) > self.config.evidence_history_length:
                raise ValueError("saved policy-change outcome history is too long")
            for outcome_state in outcomes:
                evidence.outcomes.append(
                    AutonomousOutcome.from_state_dict(outcome_state)
                )
            if self.config.use_policy_drift and (
                evidence.last_autonomous_step is None
            ) != (evidence.reference_signature is None):
                raise ValueError("saved regional evidence is internally inconsistent")
            restored_regions.append(evidence)

        window_index = state.get("budget_window_index")
        spent = int(state.get("budget_spent_steps", 0))
        if window_index is not None:
            window_index = int(window_index)
        if spent < 0 or spent > self.config.budget_steps:
            raise ValueError("saved policy-change Probe budget is invalid")
        self._regions = restored_regions
        self._budget_window_index = window_index
        self._budget_spent_steps = spent
        # A saved in-flight reservation has already charged the full horizon.
        # It is never resumed after actor restart.
        self._active_reservation = None
        self._active_session = None


__all__ = [
    "AutonomousOutcome",
    "POLICY_CHANGE_PROBE_OUTCOME_FEATURE_DIM",
    "POLICY_CHANGE_PROBE_SCHEDULER_STATE_SCHEMA_VERSION",
    "POLICY_CHANGE_PROBE_STATE_SCHEMA_VERSION",
    "PolicyChangeProbeConfig",
    "PolicyChangeProbeController",
    "PolicyChangeProbeDecision",
    "PolicyDrift",
    "PolicySignature",
    "RegionalAnchorSet",
    "policy_signature_symmetric_kl",
]
