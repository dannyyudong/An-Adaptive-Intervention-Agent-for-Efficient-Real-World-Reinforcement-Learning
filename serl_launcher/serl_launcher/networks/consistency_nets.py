"""Consistency-policy networks for ConRFT (Octo VLA backbone).

This module is part of the ConRFT integration and is imported ONLY by the
ConRFT training path. It pulls in `octo` (via the typing import) so it must
never be imported by the existing UR HIL-SERL code.

Ported from cccedric/conrft (`networks/actor_critic_nets.py` and
`networks/mlp.py`), with the action / observation encoder kept generic so the
camera obs-keys can be configured per task.
"""
from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from octo.utils.typing import Data

from serl_launcher.common.common import default_init
from serl_launcher.utils.jax_utils import (
    append_dims,
    append_zero,
    extend_and_repeat,
)


# ---------------------------------------------------------------------------
# Time embedding for the consistency model.
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, time):
        half_dim = self.dim // 2
        embeddings = jnp.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings
        return jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)


class timeMLP(nn.Module):
    t_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        activations = self.activations
        if isinstance(activations, str):
            activations = getattr(nn, activations)

        t = SinusoidalPosEmb(self.t_dim)(t)
        t = nn.Dense(self.t_dim * 2, kernel_init=default_init())(t)
        t = activations(t)
        t = nn.Dense(self.t_dim, kernel_init=default_init())(t)

        return t


# ---------------------------------------------------------------------------
# Consistency policy with an Octo-transformer based encoder.
# ---------------------------------------------------------------------------
class ConsistencyPolicy_octo(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    t_network: nn.Module
    action_dim: int
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    steps: int = 40
    clip_denoised: bool = True

    def setup(self):
        self.sigmas = self.get_sigmas_karras(
            self.steps, self.sigma_min, self.sigma_max, self.rho
        )

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = jnp.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas)

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def base_network(
        self,
        x_t: jnp.ndarray,
        sigmas: jnp.ndarray,
        obs_enc: jnp.ndarray,
        repeat: int = -1,
        train: bool = False,
    ) -> jnp.ndarray:
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim)
            for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * jnp.log(sigmas + 1e-44)

        t = self.t_network(rescaled_t)
        cont_axis = 1
        if repeat > 1:
            t = extend_and_repeat(t, 1, repeat)
            cont_axis = 2

        outputs = self.network(
            jnp.concatenate([c_in * x_t, t, obs_enc], axis=cont_axis), train=train
        )

        denoised = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        denoised = c_out * denoised + c_skip * x_t

        return denoised

    def get_features(self, observations):
        return self.encoder(observations, stop_gradient=True)

    @nn.compact
    def __call__(
        self,
        tasks: Data,
        observations: jnp.ndarray,
        action_embeddings: jnp.ndarray = None,
        x_t: jnp.ndarray = None,
        sigmas: jnp.ndarray = None,
        repeat: int = -1,
        train: bool = False,
        stop_octo_gradient: bool = True,
    ) -> jnp.ndarray:
        assert self.encoder is not None
        obs_enc, action_embeddings = self.encoder(
            observations,
            tasks=tasks,
            action_embeddings=action_embeddings,
            train=False,
            stop_gradient=stop_octo_gradient,
        )

        if obs_enc.ndim == 1:
            obs_enc = jnp.expand_dims(obs_enc, axis=0)

        if repeat > 1:
            obs_enc = extend_and_repeat(obs_enc, 1, repeat)

        if x_t is None and sigmas is None:
            batch_size = obs_enc.shape[0]
            x_shape = (
                (batch_size, repeat, self.action_dim)
                if repeat > 1
                else (batch_size, self.action_dim)
            )
            x_T = (
                jax.random.normal(self.make_rng("noise"), shape=x_shape)
                * self.sigma_max
            )
            s_in = jnp.ones((batch_size,), dtype=x_T.dtype)
            x_0 = self.base_network(x_T, self.sigmas[0] * s_in, obs_enc, repeat, train)
        else:
            x_0 = self.base_network(x_t, sigmas, obs_enc, repeat, train)

        if self.clip_denoised:
            x_0 = jnp.clip(x_0, -1, 1)

        return x_0, action_embeddings
