"""Trajectory post-processing helpers for ConRFT.

Adds Monte-Carlo returns (for Cal-QL) and Octo action embeddings to recorded
trajectories. Ported from cccedric/conrft (`examples/data_util.py`), with the
camera observation keys passed in explicitly (instead of being hard-coded to
the Franka `side_policy_256` / `wrist_1` keys) so they can match arbitrary UR
camera names.
"""
import numpy as np


def calc_return_to_go(
    rewards, terminals, gamma, reward_scale, reward_bias, reward_neg, is_sparse_reward
):
    """Compute discounted return-to-go for a single trajectory."""
    if len(rewards) == 0:
        return np.array([])

    if is_sparse_reward:
        reward_neg = reward_neg * reward_scale + reward_bias
    else:
        assert not is_sparse_reward, (
            "If you want to try on a sparse reward env, please add the reward_neg "
            "value in the ENV_CONFIG dict."
        )

    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        # All-negative sparse-reward trajectory: use r / (1 - gamma) as return-to-go.
        return_to_go = [float(reward_neg / (1 - gamma))] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * (
                1 - terminals[-i - 1]
            )
            prev_return = return_to_go[-i - 1]

    return np.array(return_to_go, dtype=np.float32)


def add_mc_returns_to_trajectory(
    trajectory, gamma, reward_scale, reward_bias, reward_neg, is_sparse_reward
):
    """Add an ``mc_returns`` field to every transition in the trajectory."""
    rewards = [t["rewards"] for t in trajectory]
    terminals = [t["dones"] for t in trajectory]

    mc_returns = calc_return_to_go(
        rewards=rewards,
        terminals=terminals,
        gamma=gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg,
        is_sparse_reward=is_sparse_reward,
    )

    for i, transition in enumerate(trajectory):
        transition["mc_returns"] = mc_returns[i]

    return trajectory


def _resize_stack(images, size):
    """Resize a (T, H, W, C) uint8 stack to (T, size[0], size[1], C). No-op if matching."""
    import cv2

    if images.shape[1:3] == tuple(size):
        return images
    out = np.stack(
        [
            cv2.resize(frame, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
            for frame in images
        ],
        axis=0,
    )
    return out.astype(np.uint8)


def add_embeddings_to_trajectory(
    trajectory,
    model,
    tasks,
    primary_image_key="side_policy_256",
    wrist_image_key="wrist_1",
    primary_image_size=(256, 256),
    wrist_image_size=(128, 128),
):
    """Add an ``embeddings`` field (Octo action embedding) to every transition.

    Args:
        trajectory: list of transition dicts.
        model: a loaded Octo model exposing ``sample_transformer``.
        tasks: Octo tasks dict (from ``model.create_tasks``).
        primary_image_key / wrist_image_key: obs keys for the primary / wrist
            cameras (mapped to Octo's ``image_primary`` / ``image_wrist``).
        primary_image_size / wrist_image_size: Octo input sizes; images are
            resized to these (Octo-small expects 256x256 primary / 128x128 wrist).
    """
    for i in range(len(trajectory)):
        observation = trajectory[i]["observations"]

        image_primary = _resize_stack(
            observation[primary_image_key], primary_image_size
        )
        image_wrist = _resize_stack(observation[wrist_image_key], wrist_image_size)
        # Add batch dimension
        image_primary = image_primary[np.newaxis, ...]
        image_wrist = image_wrist[np.newaxis, ...]
        timestep_pad_mask = np.array([[True, True]])

        observation = {
            "image_primary": image_primary,
            "image_wrist": image_wrist,
            "timestep_pad_mask": timestep_pad_mask,
        }

        action_embeddings = model.sample_transformer(observation, tasks)
        # action_embeddings is (batch_size, window_size, embedding_size)

        # remove window_size dimension (keep last timestep)
        action_embeddings = action_embeddings[:, -1, :]

        trajectory[i]["embeddings"] = action_embeddings

    return trajectory


def add_next_embeddings_to_trajectory(trajectory):
    """Add a ``next_embeddings`` field to every transition in the trajectory."""
    for i in range(len(trajectory)):
        if i == len(trajectory) - 1:
            trajectory[i]["next_embeddings"] = trajectory[i]["embeddings"]
        else:
            trajectory[i]["next_embeddings"] = trajectory[i + 1]["embeddings"]

    return trajectory
