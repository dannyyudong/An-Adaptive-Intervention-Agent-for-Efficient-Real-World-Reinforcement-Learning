"""Stable HIL-SERL observation features for the high-level scheduler."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Deque, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze
from scipy.spatial.transform import Rotation as R

from serl_launcher.aia.options import OptionID, OptionResult


FROZEN_ENCODER_SCHEMA_VERSION = "hilserl_frozen_actor_features_v1"
SCHEDULER_STATE_SCHEMA_VERSION = "hilserl_option_history_v2"
OPTION_HISTORY_ITEM_DIM = 17


@partial(jax.jit, static_argnames=("apply_fn",))
def _apply_frozen_encoder(
    params: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    apply_fn: Callable[..., Any],
) -> Any:
    """Run the complete frozen encoder as one cached XLA executable."""

    return apply_fn(
        {"params": params},
        observation,
        name="actor",
        train=False,
        return_features=True,
    )


@dataclass(frozen=True)
class FrozenObservationEncoder:
    """Immutable snapshot of the observation encoder used by the RL actor.

    Only the actor encoder subtree is retained.  Subsequent learner updates
    replace the live agent parameter tree and therefore cannot change this
    snapshot.  The returned feature is the encoder output before the actor MLP.
    """

    apply_fn: Callable[..., Any] = field(repr=False, compare=False)
    params: Mapping[str, Any] = field(repr=False, compare=False)
    feature_dim: int
    source_step: int
    schema_version: str = FROZEN_ENCODER_SCHEMA_VERSION

    @classmethod
    def from_agent(
        cls,
        agent: Any,
        sample_observation: Mapping[str, Any],
        *,
        expected_feature_dim: Optional[int] = None,
    ) -> "FrozenObservationEncoder":
        """Snapshot an agent encoder and infer/validate its feature dimension."""

        try:
            actor_encoder_params = agent.state.params["modules_actor"]["encoder"]
            apply_fn = agent.state.apply_fn
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError(
                "Agent does not expose params['modules_actor']['encoder']"
            ) from exc

        # JAX arrays are immutable.  Rebuilding the pytree detaches this object
        # from later replacements of the live agent parameter tree while
        # retaining only the parameters required by Policy(return_features=True).
        encoder_params = jax.tree.map(jnp.array, actor_encoder_params)
        frozen_params = freeze({"modules_actor": {"encoder": encoder_params}})
        source_step = int(np.asarray(jax.device_get(agent.state.step)))

        provisional = cls(
            apply_fn=apply_fn,
            params=frozen_params,
            feature_dim=-1,
            source_step=source_step,
        )
        # This first synchronous validation also compiles and warms the jitted
        # encoder before the real-time actor loop starts.
        sample_features = provisional._encode_unchecked(sample_observation)
        feature_dim = provisional._validate_single_features(
            sample_features,
            expected_feature_dim=expected_feature_dim,
        ).shape[0]

        return cls(
            apply_fn=apply_fn,
            params=frozen_params,
            feature_dim=int(feature_dim),
            source_step=source_step,
        )

    def _encode_unchecked(self, observation: Mapping[str, Any]) -> Any:
        return _apply_frozen_encoder(
            self.params,
            observation,
            apply_fn=self.apply_fn,
        )

    @staticmethod
    def _validate_single_features(
        features: Any,
        *,
        expected_feature_dim: Optional[int],
    ) -> np.ndarray:
        array = np.asarray(jax.device_get(features), dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(
                "Frozen observation encoder must produce one flat feature vector; "
                f"got shape {array.shape}"
            )
        if expected_feature_dim is not None and array.shape[0] != int(
            expected_feature_dim
        ):
            raise ValueError(
                "Frozen observation encoder feature dimension mismatch: "
                f"expected {expected_feature_dim}, got {array.shape[0]}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("Frozen observation encoder produced NaN or Inf")
        return array

    def encode(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Encode one actor observation into a stable flat float32 vector."""

        features = self._encode_unchecked(observation)
        return self._validate_single_features(
            features,
            expected_feature_dim=self.feature_dim,
        )


def _positive_scale(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return value


def _extract_serl_proprio(observation: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    """Return force, xyz, and MRP from fixed- or learned-gripper state."""

    try:
        state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
    except (KeyError, TypeError) as exc:
        raise ValueError("Observation does not contain a flattened 'state'") from exc
    if not np.all(np.isfinite(state)):
        raise ValueError(
            "Scheduler motion history expects finite "
            "tcp_force(3) + tcp_pose(6), optionally preceded by "
            f"gripper_state(2); got {state.shape}"
        )
    if state.shape == (9,):
        force = state[:3]
        tcp_pose = state[3:9]
    elif state.shape == (11,):
        force = state[2:5]
        tcp_pose = state[5:11]
    else:
        raise ValueError(
            "Scheduler motion history expects finite "
            "tcp_force(3) + tcp_pose(6), optionally preceded by "
            f"gripper_state(2); got {state.shape}"
        )
    return force.copy(), tcp_pose[:3].copy(), tcp_pose[3:6].copy()


@dataclass(frozen=True)
class OptionHistoryConfig:
    length: int = 4
    max_duration: int = 200
    reward_scale: float = 1.0
    position_scale_m: float = 0.05
    rotation_scale_rad: float = 0.2
    path_length_scale_m: float = 0.10
    force_scale_n: float = 80.0

    def __post_init__(self) -> None:
        if int(self.length) <= 0:
            raise ValueError("Option history length must be positive")
        if int(self.max_duration) <= 0:
            raise ValueError("Option history max_duration must be positive")
        for name in (
            "reward_scale",
            "position_scale_m",
            "rotation_scale_rad",
            "path_length_scale_m",
            "force_scale_n",
        ):
            _positive_scale(getattr(self, name), name)


@dataclass(frozen=True)
class OptionMotionSummary:
    delta_xyz: np.ndarray
    delta_rotvec: np.ndarray
    path_length_m: float
    delta_force: np.ndarray
    mean_force_magnitude_n: float
    max_force_magnitude_n: float


class OptionMotionAccumulator:
    """Accumulate compact robot-motion statistics over one active Option."""

    def __init__(self, start_observation: Mapping[str, Any]) -> None:
        force, xyz, mrp = _extract_serl_proprio(start_observation)
        self._start_force = force
        self._start_xyz = xyz
        self._start_mrp = mrp
        self._last_force = force
        self._last_xyz = xyz
        self._last_mrp = mrp
        self._path_length_m = 0.0
        self._force_magnitude_sum = 0.0
        self._max_force_magnitude_n = 0.0
        self._steps = 0

    def observe(self, observation: Mapping[str, Any]) -> None:
        force, xyz, mrp = _extract_serl_proprio(observation)
        self._path_length_m += float(np.linalg.norm(xyz - self._last_xyz))
        force_magnitude = float(np.linalg.norm(force))
        self._force_magnitude_sum += force_magnitude
        self._max_force_magnitude_n = max(self._max_force_magnitude_n, force_magnitude)
        self._last_force = force
        self._last_xyz = xyz
        self._last_mrp = mrp
        self._steps += 1

    def finish(self) -> OptionMotionSummary:
        relative_rotation = (
            R.from_mrp(self._last_mrp) * R.from_mrp(self._start_mrp).inv()
        )
        mean_force = self._force_magnitude_sum / max(self._steps, 1)
        return OptionMotionSummary(
            delta_xyz=(self._last_xyz - self._start_xyz).astype(np.float32),
            delta_rotvec=relative_rotation.as_rotvec().astype(np.float32),
            path_length_m=float(self._path_length_m),
            delta_force=(self._last_force - self._start_force).astype(np.float32),
            mean_force_magnitude_n=float(mean_force),
            max_force_magnitude_n=float(self._max_force_magnitude_n),
        )


class OptionHistory:
    """Fixed-length FIFO history of completed Option outcomes and motion."""

    def __init__(self, config: OptionHistoryConfig) -> None:
        self.config = config
        self._items: Deque[np.ndarray] = deque(maxlen=int(config.length))

    def __len__(self) -> int:
        return len(self._items)

    def reset(self) -> None:
        self._items.clear()

    def append(
        self,
        result: OptionResult,
        scheduler_reward: float,
        motion: OptionMotionSummary,
    ) -> None:
        option_one_hot = np.zeros(len(OptionID), dtype=np.float32)
        option_one_hot[int(result.option_id)] = 1.0
        duration = np.array(
            [min(float(result.duration) / float(self.config.max_duration), 1.0)],
            dtype=np.float32,
        )
        reward = np.array(
            [float(scheduler_reward) / float(self.config.reward_scale)],
            dtype=np.float32,
        )
        delta_xyz = np.clip(
            np.asarray(motion.delta_xyz, dtype=np.float32)
            / float(self.config.position_scale_m),
            -1.0,
            1.0,
        )
        delta_rotvec = np.clip(
            np.asarray(motion.delta_rotvec, dtype=np.float32)
            / float(self.config.rotation_scale_rad),
            -1.0,
            1.0,
        )
        path_length = np.array(
            [
                min(
                    float(motion.path_length_m)
                    / float(self.config.path_length_scale_m),
                    1.0,
                )
            ],
            dtype=np.float32,
        )
        delta_force = np.clip(
            np.asarray(motion.delta_force, dtype=np.float32)
            / float(self.config.force_scale_n),
            -1.0,
            1.0,
        )
        force_statistics = np.array(
            [
                min(
                    float(motion.mean_force_magnitude_n)
                    / float(self.config.force_scale_n),
                    1.0,
                ),
                min(
                    float(motion.max_force_magnitude_n)
                    / float(self.config.force_scale_n),
                    1.0,
                ),
            ],
            dtype=np.float32,
        )
        item = np.concatenate(
            (
                option_one_hot,
                duration,
                reward,
                delta_xyz,
                delta_rotvec,
                path_length,
                delta_force,
                force_statistics,
            )
        ).astype(np.float32)
        if item.shape != (OPTION_HISTORY_ITEM_DIM,) or not np.all(np.isfinite(item)):
            raise ValueError(f"Invalid Option history item with shape {item.shape}")
        self._items.append(item)

    def encode(self) -> tuple[np.ndarray, np.ndarray]:
        history = np.zeros(
            (int(self.config.length), OPTION_HISTORY_ITEM_DIM), dtype=np.float32
        )
        mask = np.zeros((int(self.config.length),), dtype=np.float32)
        if self._items:
            count = len(self._items)
            history[-count:] = np.stack(tuple(self._items), axis=0)
            mask[-count:] = 1.0
        return history, mask


@dataclass(frozen=True)
class SchedulerStateBuilder:
    """Combine frozen RL features with completed-Option history."""

    encoder: FrozenObservationEncoder
    history_config: OptionHistoryConfig
    extra_feature_dim: int = 0
    extra_feature_fn: Optional[Callable[[Mapping[str, Any]], np.ndarray]] = field(
        default=None,
        repr=False,
        compare=False,
    )
    schema_version: str = SCHEDULER_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        extra_feature_dim = int(self.extra_feature_dim)
        if extra_feature_dim < 0:
            raise ValueError("extra_feature_dim must be non-negative")
        if (extra_feature_dim == 0) != (self.extra_feature_fn is None):
            raise ValueError(
                "extra_feature_fn must be provided exactly when "
                "extra_feature_dim is positive"
            )

    @property
    def state_dim(self) -> int:
        return (
            self.encoder.feature_dim
            + int(self.history_config.length) * (OPTION_HISTORY_ITEM_DIM + 1)
            + int(self.extra_feature_dim)
        )

    def build(
        self,
        observation: Mapping[str, Any],
        history: OptionHistory,
    ) -> np.ndarray:
        if history.config != self.history_config:
            raise ValueError(
                "OptionHistory config does not match SchedulerStateBuilder"
            )
        encoded_history, history_mask = history.encode()
        extra_features = np.zeros((0,), dtype=np.float32)
        if self.extra_feature_fn is not None:
            extra_features = np.asarray(
                self.extra_feature_fn(observation), dtype=np.float32
            )
            if extra_features.shape != (int(self.extra_feature_dim),) or not np.all(
                np.isfinite(extra_features)
            ):
                raise ValueError(
                    "Invalid Scheduler extra features with shape "
                    f"{extra_features.shape}"
                )
        state = np.concatenate(
            (
                self.encoder.encode(observation),
                encoded_history.reshape(-1),
                history_mask,
                extra_features,
            )
        ).astype(np.float32)
        if state.shape != (self.state_dim,) or not np.all(np.isfinite(state)):
            raise ValueError(f"Invalid Scheduler state with shape {state.shape}")
        return state
