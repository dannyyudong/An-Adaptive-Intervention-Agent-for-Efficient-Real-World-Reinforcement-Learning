"""Launcher helper for the ConRFT (Octo VLA + consistency policy) agent.

Kept separate from `serl_launcher.utils.launcher` so the existing UR launcher
stays free of any `octo` import. This module is imported ONLY by the ConRFT
training path.
"""
from jax import nn

from serl_launcher.agents.continuous.conrft_single_octo_cp import (
    ConrftCPOctoAgentSingleArm,
)
from serl_launcher.utils.launcher import make_batch_augmentation_func


def make_conrft_octo_cp_pixel_agent_single_arm(
    seed,
    sample_obs,
    sample_action,
    sample_tasks,
    octo_model,
    encoder_type="resnet-pretrained",
    image_keys=("image",),
    primary_image_key="side_policy_256",
    wrist_image_key="wrist_1",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    num_scales=40,
    sigma_data: float = 0.5,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    fix_gripper: bool = False,
    q_weight: float = 0.1,
    bc_weight: float = 1.0,
    cql_n_actions: int = 10,
):
    import jax

    agent = ConrftCPOctoAgentSingleArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        sample_tasks,
        encoder_type=encoder_type,
        use_proprio=True,
        octo_model=octo_model,
        image_keys=image_keys,
        primary_image_key=primary_image_key,
        wrist_image_key=wrist_image_key,
        fix_gripper=fix_gripper,
        policy_kwargs={
            "sigma_data": sigma_data,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "rho": rho,
            "steps": num_scales,
            "clip_denoised": True,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_t_network_kwargs={
            "t_dim": 16,
            "activations": nn.tanh,
        },
        num_scales=num_scales,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
        rho=rho,
        discount=discount,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        cql_n_actions=cql_n_actions,
        augmentation_function=make_batch_augmentation_func(image_keys),
        q_weight=q_weight,
        bc_weight=bc_weight,
    )
    return agent
