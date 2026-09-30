"""Discrete Double DQN learner for the Option-level Scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from serl_launcher.common.common import JaxRLTrainState, nonpytree_field
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.networks.mlp import MLP


@dataclass(frozen=True)
class SchedulerDQNConfig:
    """Algorithm defaults; experiments may override task-sensitive values."""

    hidden_dims: tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4
    batch_size: int = 64
    warmup_transitions: int = 256
    updates_per_transition: int = 1
    max_updates_per_loop: int = 16
    recent_sample_fraction: float = 0.80
    recent_sample_window: int = 5_000
    target_update_tau: float = 0.005
    gradient_clip: float = 10.0
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000
    exploration_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(int(dim) <= 0 for dim in self.hidden_dims):
            raise ValueError("Scheduler hidden_dims must contain positive dimensions")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("Scheduler learning_rate must be positive")
        if int(self.batch_size) <= 0:
            raise ValueError("Scheduler batch_size must be positive")
        if int(self.warmup_transitions) < int(self.batch_size):
            raise ValueError("Scheduler warmup_transitions must be at least batch_size")
        if int(self.updates_per_transition) <= 0:
            raise ValueError("Scheduler updates_per_transition must be positive")
        if int(self.max_updates_per_loop) <= 0:
            raise ValueError("Scheduler max_updates_per_loop must be positive")
        if not 0.0 <= float(self.recent_sample_fraction) <= 1.0:
            raise ValueError("Scheduler recent_sample_fraction must be in [0, 1]")
        if int(self.recent_sample_window) <= 0:
            raise ValueError("Scheduler recent_sample_window must be positive")
        if not 0.0 < float(self.target_update_tau) <= 1.0:
            raise ValueError("Scheduler target_update_tau must be in (0, 1]")
        if not np.isfinite(self.gradient_clip) or self.gradient_clip <= 0.0:
            raise ValueError("Scheduler gradient_clip must be positive")
        if not 0.0 <= float(self.epsilon_end) <= float(self.epsilon_start) <= 1.0:
            raise ValueError(
                "Scheduler epsilon values must satisfy 0 <= end <= start <= 1"
            )
        if int(self.epsilon_decay_steps) <= 0:
            raise ValueError("Scheduler epsilon_decay_steps must be positive")
        if self.exploration_weights is not None:
            exploration_weights = np.asarray(self.exploration_weights, dtype=np.float64)
            if exploration_weights.ndim != 1 or exploration_weights.size == 0:
                raise ValueError(
                    "Scheduler exploration_weights must be a non-empty vector"
                )
            if not np.all(np.isfinite(exploration_weights)) or np.any(
                exploration_weights <= 0.0
            ):
                raise ValueError(
                    "Scheduler exploration_weights must be finite and positive"
                )

    def epsilon(self, scheduler_step: int) -> float:
        fraction = min(max(float(scheduler_step), 0.0) / self.epsilon_decay_steps, 1.0)
        return float(
            self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
        )


class SchedulerQNetwork(nn.Module):
    num_options: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        return MLP(
            hidden_dims=(*tuple(self.hidden_dims), int(self.num_options)),
            activate_final=False,
        )(observations, train=False)


def masked_double_dqn_targets(
    *,
    rewards: jax.Array,
    masks: jax.Array,
    discounts: jax.Array,
    online_next_q: jax.Array,
    target_next_q: jax.Array,
    next_available_actions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute SMDP Double DQN targets and selected next actions."""
    invalid_q = jnp.finfo(online_next_q.dtype).min
    masked_online_next_q = jnp.where(next_available_actions, online_next_q, invalid_q)
    next_actions = jnp.argmax(masked_online_next_q, axis=-1)
    bootstrap_q = jnp.take_along_axis(
        target_next_q, next_actions[..., None], axis=-1
    ).squeeze(-1)
    bootstrap_q = jnp.where(masks > 0.0, bootstrap_q, 0.0)
    targets = rewards + masks * discounts * bootstrap_q
    return jax.lax.stop_gradient(targets), next_actions


class SchedulerDQNAgent(flax.struct.PyTreeNode):
    """Independent Double DQN state and optimizer for high-level Options."""

    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def q_values(
        self,
        observations: jax.Array,
        *,
        target: bool = False,
    ) -> jax.Array:
        params = self.state.target_params if target else self.state.params
        return self.state.apply_fn({"params": params}, observations)

    @partial(jax.jit, static_argnames=())
    def update(self, batch: Batch) -> tuple["SchedulerDQNAgent", dict]:
        """Apply one masked SMDP Double DQN update."""

        def loss_fn(params):
            online_next_q = self.state.apply_fn(
                {"params": params}, batch["next_observations"]
            )
            target_next_q_all = self.state.apply_fn(
                {"params": self.state.target_params}, batch["next_observations"]
            )
            targets, _ = masked_double_dqn_targets(
                rewards=batch["rewards"],
                masks=batch["masks"],
                discounts=batch["discounts"],
                online_next_q=online_next_q,
                target_next_q=target_next_q_all,
                next_available_actions=batch["next_available_actions"],
            )

            q_all = self.state.apply_fn({"params": params}, batch["observations"])
            selected_q = jnp.take_along_axis(
                q_all, batch["actions"].astype(jnp.int32)[..., None], axis=-1
            ).squeeze(-1)
            td_error = selected_q - targets
            loss = jnp.mean(optax.huber_loss(selected_q, targets))
            info = {
                "loss": loss,
                "q_mean": jnp.mean(q_all),
                "q_selected_mean": jnp.mean(selected_q),
                "target_mean": jnp.mean(targets),
                "td_error_mean": jnp.mean(td_error),
                "abs_td_error_mean": jnp.mean(jnp.abs(td_error)),
                "reward_mean": jnp.mean(batch["rewards"]),
                "discount_mean": jnp.mean(batch["discounts"]),
                "duration_mean": jnp.mean(batch["durations"]),
                "q_rl_mean": jnp.mean(q_all[..., 0]),
                "q_trajectory_mean": jnp.mean(q_all[..., 1]),
                "q_code_mean": jnp.mean(q_all[..., 2]),
            }
            return loss, info

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.state.params)
        info["grad_norm"] = optax.global_norm(grads)
        new_state = self.state.apply_gradients(grads=grads)
        new_state = new_state.target_update(self.config["target_update_tau"])
        return self.replace(state=new_state), info

    def sample_action(
        self,
        observation: np.ndarray,
        available_actions: np.ndarray,
        *,
        seed: PRNGKey,
        epsilon: float,
    ) -> tuple[int, float, np.ndarray]:
        """Masked epsilon-greedy action and exact behavior probability."""
        action_mask = np.asarray(available_actions, dtype=bool)
        if action_mask.shape != (int(self.config["num_options"]),):
            raise ValueError(
                "Scheduler action mask has shape "
                f"{action_mask.shape}, expected {(self.config['num_options'],)}"
            )
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            raise ValueError("Scheduler action mask contains no valid Option")
        configured_exploration_weights = self.config.get("exploration_weights")
        if configured_exploration_weights is None:
            exploration_probabilities = np.zeros_like(action_mask, dtype=np.float64)
            exploration_probabilities[valid_actions] = 1.0 / float(valid_actions.size)
        else:
            exploration_weights = np.asarray(
                configured_exploration_weights, dtype=np.float64
            )
            if exploration_weights.shape != action_mask.shape:
                raise ValueError(
                    "Scheduler exploration_weights has shape "
                    f"{exploration_weights.shape}, expected {action_mask.shape}"
                )
            masked_weights = np.where(action_mask, exploration_weights, 0.0)
            exploration_probabilities = masked_weights / float(np.sum(masked_weights))
        epsilon = float(np.clip(epsilon, 0.0, 1.0))
        q_values = np.asarray(
            jax.device_get(self.q_values(jnp.asarray(observation, dtype=jnp.float32))),
            dtype=np.float32,
        )
        masked_q = np.where(action_mask, q_values, -np.inf)
        greedy_action = int(np.argmax(masked_q))
        explore_key, action_key = jax.random.split(seed)
        explore = bool(
            np.asarray(jax.device_get(jax.random.bernoulli(explore_key, epsilon)))
        )
        if explore:
            if configured_exploration_weights is None:
                index = int(
                    np.asarray(
                        jax.device_get(
                            jax.random.randint(action_key, (), 0, valid_actions.size)
                        )
                    )
                )
                action = int(valid_actions[index])
            else:
                action = int(
                    np.asarray(
                        jax.device_get(
                            jax.random.choice(
                                action_key,
                                int(self.config["num_options"]),
                                p=jnp.asarray(
                                    exploration_probabilities,
                                    dtype=jnp.float32,
                                ),
                            )
                        )
                    )
                )
        else:
            action = greedy_action
        behavior_prob = epsilon * float(exploration_probabilities[action])
        if action == greedy_action:
            behavior_prob += 1.0 - epsilon
        return action, float(behavior_prob), q_values

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        *,
        state_dim: int,
        num_options: int,
        config: SchedulerDQNConfig,
    ) -> "SchedulerDQNAgent":
        if int(state_dim) <= 0 or int(num_options) <= 0:
            raise ValueError("Scheduler state_dim and num_options must be positive")
        if config.exploration_weights is not None and len(
            config.exploration_weights
        ) != int(num_options):
            raise ValueError(
                "Scheduler exploration_weights must contain one weight per "
                f"Option; got {len(config.exploration_weights)} weights for "
                f"{int(num_options)} Options"
            )
        network = SchedulerQNetwork(
            num_options=int(num_options), hidden_dims=tuple(config.hidden_dims)
        )
        rng, init_rng, state_rng = jax.random.split(rng, 3)
        params = network.init(
            init_rng, jnp.zeros((int(state_dim),), dtype=jnp.float32)
        )["params"]
        optimizer = make_optimizer(
            learning_rate=float(config.learning_rate),
            clip_grad_norm=float(config.gradient_clip),
        )
        state = JaxRLTrainState.create(
            apply_fn=network.apply,
            params=params,
            target_params=params,
            txs=optimizer,
            rng=state_rng,
            epsilon=float(config.epsilon_start),
        )
        return cls(
            state=state,
            config={
                "state_dim": int(state_dim),
                "num_options": int(num_options),
                "target_update_tau": float(config.target_update_tau),
                "exploration_weights": (
                    None
                    if config.exploration_weights is None
                    else tuple(float(weight) for weight in config.exploration_weights)
                ),
            },
        )
