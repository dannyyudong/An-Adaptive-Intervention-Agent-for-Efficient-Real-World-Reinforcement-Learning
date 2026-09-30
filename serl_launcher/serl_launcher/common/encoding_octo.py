"""Octo-transformer observation encoder for ConRFT.

Part of the ConRFT integration; imported ONLY by the ConRFT training path
(it depends on `octo`). Must never be imported by existing UR HIL-SERL code.

Ported from cccedric/conrft (`common/encoding.py::OctoEncodingWrapper`), with
the camera observation keys made configurable (`primary_image_key` /
`wrist_image_key`) instead of being hard-coded to the Franka `side_policy_256`
/ `wrist_1` keys, so it can be wired to arbitrary UR camera names.
"""
from typing import Dict, Iterable

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from octo.model.octo_module import OctoTransformer
from octo.utils.typing import Data


def _resize_image_stack(images: jnp.ndarray, size) -> jnp.ndarray:
    """Resize a (B, T, H, W, C) uint8 image stack to (B, T, size[0], size[1], C).

    No-op when the spatial dims already match. Returns uint8 in [0, 255].
    """
    b, t, h, w, c = images.shape
    if (h, w) == tuple(size):
        return images
    x = images.reshape(b * t, h, w, c).astype(jnp.float32)
    x = jax.image.resize(x, (b * t, size[0], size[1], c), method="bilinear")
    x = jnp.clip(jnp.round(x), 0, 255).astype(jnp.uint8)
    return x.reshape(b, t, size[0], size[1], c)


class LastFrameEncodingWrapper(nn.Module):
    """ResNet image encoder used by the ConRFT critic.

    Same as the shared `serl_launcher.common.encoding.EncodingWrapper`, but when
    observations are frame-stacked (T>1) it keeps only the LAST frame before
    encoding. ConRFT runs with a window of 2 frames (for the Octo actor), while
    the frozen pretrained ResNet critic expects a single 3-channel frame. Kept
    here (not in the shared module) so existing UR agents are unaffected.

    Ported from cccedric/conrft (`common/encoding.py::EncodingWrapper`).
    """

    encoder: Dict[str, nn.Module]
    use_proprio: bool
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    image_keys: Iterable[str] = ("image",)

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train=False,
        stop_gradient=False,
        is_encoded=False,
    ) -> jnp.ndarray:
        encoded = []
        for image_key in self.image_keys:
            image = observations[image_key]
            if not is_encoded:
                if self.enable_stacking:
                    if len(image.shape) == 4:
                        T = image.shape[0]
                        if T > 1:
                            image = image[-1:]  # only the last frame
                        image = rearrange(image, "T H W C -> H W (T C)")
                    if len(image.shape) == 5:
                        T = image.shape[1]
                        if T > 1:
                            image = image[:, -1:]
                        image = rearrange(image, "B T H W C -> B H W (T C)")

            image = self.encoder[image_key](image, train=train, encode=not is_encoded)
            if stop_gradient:
                image = jax.lax.stop_gradient(image)
            encoded.append(image)

        encoded = jnp.concatenate(encoded, axis=-1)

        if self.use_proprio:
            state = observations["state"]
            if self.enable_stacking:
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                    encoded = encoded.reshape(-1)
                if len(state.shape) == 3:
                    state = rearrange(state, "B T C -> B (T C)")
            state = nn.Dense(
                self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform()
            )(state)
            state = nn.LayerNorm()(state)
            state = nn.tanh(state)
            encoded = jnp.concatenate([encoded, state], axis=-1)

        return encoded


class OctoEncodingWrapper(nn.Module):
    """Encodes observations with an Octo transformer into a single flat encoding.

    Args:
        encoder: The Octo transformer module.
        use_proprio: Whether to concatenate proprioception (after encoding).
        primary_image_key: obs key for the third-person / primary camera image
            (mapped to Octo's ``image_primary``).
        wrist_image_key: obs key for the wrist camera image (mapped to Octo's
            ``image_wrist``).
    """

    encoder: OctoTransformer
    use_proprio: bool
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    image_keys: Iterable[str] = ("image",)
    primary_image_key: str = "side_policy_256"
    wrist_image_key: str = "wrist_1"
    mask_wrist_prob: float = 0.2
    # Octo-small expects image_primary 256x256 and image_wrist 128x128. UR cameras
    # come out at 128x128, so the primary view is resized up to match.
    primary_image_size: tuple = (256, 256)
    wrist_image_size: tuple = (128, 128)

    @nn.compact
    def __call__(
        self,
        observations: Data,
        tasks: Data = None,
        action_embeddings: jnp.ndarray = None,
        train: bool = True,
        stop_gradient: bool = False,
    ) -> jnp.ndarray:
        if action_embeddings is None:
            image_primary = observations[self.primary_image_key]
            image_wrist = observations[self.wrist_image_key]
            if image_primary.ndim == 4:
                image_primary = image_primary[jnp.newaxis, ...]
                image_wrist = image_wrist[jnp.newaxis, ...]
            # Resize to Octo's expected input sizes (no-op if already matching).
            image_primary = _resize_image_stack(image_primary, self.primary_image_size)
            image_wrist = _resize_image_stack(image_wrist, self.wrist_image_size)
            batch_size = image_primary.shape[0]
            window_size = image_primary.shape[1]
            timestep_pad_mask = jnp.ones((batch_size, window_size), dtype=bool)

            if not stop_gradient:

                def mask_image(image, mask_flag):
                    return jax.lax.cond(
                        mask_flag,
                        lambda _: jnp.zeros_like(image),
                        lambda _: image,
                        operand=None,
                    )

                mask_flags = jax.random.bernoulli(
                    self.make_rng("mask_wrist"),
                    p=self.mask_wrist_prob,
                    shape=(batch_size,),
                )
                image_wrist = jax.vmap(mask_image)(image_wrist, mask_flags)

            observation_octo = {
                "image_primary": image_primary,
                "image_wrist": image_wrist,
                "timestep_pad_mask": timestep_pad_mask,
            }

            transformer_outputs = self.encoder(
                observation_octo, tasks, timestep_pad_mask, train=not stop_gradient
            )
            token_group = transformer_outputs["readout_action"]
            action_embeddings = token_group.tokens.mean(axis=-2)

            # remove window_size dimension
            action_embeddings = action_embeddings[:, -1, :]
        else:
            action_embeddings = action_embeddings

        if stop_gradient:
            action_embeddings = jax.lax.stop_gradient(action_embeddings)

        encoded = action_embeddings
        if self.use_proprio:
            state = observations["state"]
            if self.enable_stacking:
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                    encoded = encoded.reshape(-1)
                if len(state.shape) == 3:
                    state = rearrange(state, "B T C -> B (T C)")
            state = nn.Dense(
                self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform()
            )(state)
            state = nn.LayerNorm()(state)
            state = nn.tanh(state)
            encoded = jnp.concatenate([encoded, state], axis=-1)

        return encoded, action_embeddings
