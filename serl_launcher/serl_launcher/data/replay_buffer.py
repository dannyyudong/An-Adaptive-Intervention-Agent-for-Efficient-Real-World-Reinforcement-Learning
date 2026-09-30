import collections
from typing import Any, Iterator, Optional, Sequence, Tuple, Union

import gymnasium as gym
import jax
import numpy as np
from serl_launcher.data.dataset import Dataset, DatasetDict


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        try:
            dataset_dict[insert_index] = data_dict
        except Exception as e:
            print(data_dict)
            raise (e)
    elif isinstance(dataset_dict, dict):
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        include_next_actions: Optional[bool] = False,
        include_label: Optional[bool] = False,
        include_grasp_penalty: Optional[bool] = False,
        include_octo_embeddings: Optional[bool] = False,
        include_mc_returns: Optional[bool] = False,
        octo_embedding_dim: int = 384,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        if include_next_actions:
            dataset_dict["next_actions"] = np.empty(
                (capacity, *action_space.shape), dtype=action_space.dtype
            )
            dataset_dict["next_intvn"] = np.empty((capacity,), dtype=bool)

        if include_label:
            dataset_dict["labels"] = np.empty((capacity,), dtype=int)

        if include_grasp_penalty:
            dataset_dict["grasp_penalty"] = np.empty((capacity,), dtype=np.float32)

        # ConRFT: Monte-Carlo returns (for Cal-QL) and Octo action embeddings.
        if include_mc_returns:
            dataset_dict["mc_returns"] = np.empty((capacity,), dtype=np.float32)

        if include_octo_embeddings:
            dataset_dict["embeddings"] = np.empty(
                (capacity, octo_embedding_dim), dtype=np.float32
            )
            dataset_dict["next_embeddings"] = np.empty(
                (capacity, octo_embedding_dim), dtype=np.float32
            )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        # for key in data_dict:
        #     print(type(data_dict[key]))
        #     if type(data_dict[key]) == dict:
        #         for key2 in data_dict[key]:
        #             print(data_dict[key][key2])
        #             print(data_dict[key][key2][0].shape())
        #     print(data_dict[key])
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}, device=None):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data, device=device))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def download(self, from_idx: int, to_idx: int):
        indices = np.arange(from_idx, to_idx)
        data_dict = self.sample(batch_size=len(indices), indx=indices)
        return to_idx, data_dict

    def get_download_iterator(self):
        last_idx = 0
        while True:
            if last_idx >= self._size:
                raise RuntimeError(f"last_idx {last_idx} >= self._size {self._size}")
            last_idx, batch = self.download(last_idx, self._size)
            yield batch


class SchedulerReplayBuffer(ReplayBuffer):
    """Fixed-schema replay buffer for Option-level SMDP transitions."""

    def __init__(self, state_dim: int, num_options: int, capacity: int):
        if int(state_dim) <= 0:
            raise ValueError("Scheduler state_dim must be positive")
        if int(num_options) <= 0:
            raise ValueError("Scheduler num_options must be positive")
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(state_dim),),
            dtype=np.float32,
        )
        action_space = gym.spaces.Discrete(int(num_options))
        super().__init__(observation_space, action_space, int(capacity))
        self.state_dim = int(state_dim)
        self.num_options = int(num_options)
        self.dataset_dict.update(
            {
                "durations": np.empty((capacity,), dtype=np.int32),
                "discounts": np.empty((capacity,), dtype=np.float32),
                "terminated": np.empty((capacity,), dtype=bool),
                "truncated": np.empty((capacity,), dtype=bool),
                "available_actions": np.empty((capacity, self.num_options), dtype=bool),
                "next_available_actions": np.empty(
                    (capacity, self.num_options), dtype=bool
                ),
                "termination_reason": np.empty((capacity,), dtype=np.int32),
                "behavior_prob": np.empty((capacity,), dtype=np.float32),
                "start_step": np.empty((capacity,), dtype=np.int64),
                "rl_policy_version": np.empty((capacity,), dtype=np.int64),
            }
        )

    def sample(
        self,
        batch_size: int,
        keys=None,
        indx=None,
        *,
        recent_fraction: float = 0.0,
        recent_window: int = 1,
    ):
        """Sample a mixture of recent and global Option transitions."""
        if indx is not None or float(recent_fraction) <= 0.0:
            return super().sample(batch_size=batch_size, keys=keys, indx=indx)
        if not 0.0 <= float(recent_fraction) <= 1.0:
            raise ValueError("recent_fraction must be in [0, 1]")
        if int(recent_window) <= 0:
            raise ValueError("recent_window must be positive")
        if self._size <= 0:
            raise ValueError("Cannot sample from an empty Scheduler replay buffer")

        recent_count = min(self._size, int(recent_window))
        if self._size < self._capacity:
            recent_indices = np.arange(
                self._size - recent_count, self._size, dtype=np.int64
            )
        else:
            recent_indices = (
                np.arange(
                    self._insert_index - recent_count,
                    self._insert_index,
                    dtype=np.int64,
                )
                % self._capacity
            )
        num_recent = int(round(int(batch_size) * float(recent_fraction)))
        num_recent = min(max(num_recent, 0), int(batch_size))
        num_global = int(batch_size) - num_recent
        sampled_indices = []
        if num_recent:
            sampled_indices.append(
                self.np_random.choice(recent_indices, size=num_recent, replace=True)
            )
        if num_global:
            if hasattr(self.np_random, "integers"):
                global_indices = self.np_random.integers(self._size, size=num_global)
            else:
                global_indices = self.np_random.randint(self._size, size=num_global)
            sampled_indices.append(global_indices)
        indices = np.concatenate(sampled_indices).astype(np.int64, copy=False)
        self.np_random.shuffle(indices)
        return super().sample(batch_size=batch_size, keys=keys, indx=indices)
