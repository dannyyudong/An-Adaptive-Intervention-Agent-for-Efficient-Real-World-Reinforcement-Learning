import copy
from collections import deque
from threading import Lock
from typing import Union, Iterable

import gymnasium as gym
import jax
from serl_launcher.data.replay_buffer import ReplayBuffer, SchedulerReplayBuffer
from serl_launcher.data.memory_efficient_replay_buffer import (
    MemoryEfficientReplayBuffer,
)

from agentlace.data.data_store import DataStoreBase


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()
        self._data_id = 0
        self._raw_data = deque(maxlen=min(capacity, 10000))

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            if args:
                self._raw_data.append((self._data_id, copy.deepcopy(args[0])))
                self._data_id += 1
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._data_id

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        with self._lock:
            from_id = max(0, int(from_id))
            return [
                copy.deepcopy(data)
                for data_id, data in self._raw_data
                if data_id >= from_id
            ]


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        **kwargs,
    ):
        MemoryEfficientReplayBuffer.__init__(
            self,
            observation_space,
            action_space,
            capacity,
            pixel_keys=image_keys,
            **kwargs,
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()
        self._data_id = 0
        self._raw_data = deque(maxlen=min(capacity, 10000))

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            if args:
                self._raw_data.append((self._data_id, copy.deepcopy(args[0])))
                self._data_id += 1
            super(MemoryEfficientReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(MemoryEfficientReplayBufferDataStore, self).sample(
                *args, **kwargs
            )

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._data_id

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        with self._lock:
            from_id = max(0, int(from_id))
            return [
                copy.deepcopy(data)
                for data_id, data in self._raw_data
                if data_id >= from_id
            ]


class SchedulerReplayBufferDataStore(SchedulerReplayBuffer, DataStoreBase):
    """Thread-safe Agentlace datastore for high-level Scheduler transitions."""

    def __init__(
        self,
        state_dim: int,
        num_options: int,
        capacity: int,
    ):
        SchedulerReplayBuffer.__init__(self, state_dim, num_options, capacity)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()
        self._data_id = 0
        self._raw_data = deque(maxlen=min(capacity, 10000))

    def insert(self, *args, **kwargs):
        with self._lock:
            if args:
                self._raw_data.append((self._data_id, copy.deepcopy(args[0])))
                self._data_id += 1
            super(SchedulerReplayBufferDataStore, self).insert(*args, **kwargs)

    def sample(self, *args, **kwargs):
        with self._lock:
            return super(SchedulerReplayBufferDataStore, self).sample(*args, **kwargs)

    def latest_data_id(self):
        return self._data_id

    def get_latest_data(self, from_id: int):
        with self._lock:
            from_id = max(0, int(from_id))
            return [
                copy.deepcopy(data)
                for data_id, data in self._raw_data
                if data_id >= from_id
            ]


def populate_data_store(
    data_store: DataStoreBase,
    demos_path: str,
):
    """
    Utility function to populate demonstrations data into data_store.
    :return data_store
    """
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                data_store.insert(transition)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store


def populate_data_store_with_z_axis_only(
    data_store: DataStoreBase,
    demos_path: str,
):
    """
    Utility function to populate demonstrations data into data_store.
    This will remove the x and y cartesian coordinates from the state.
    :return data_store
    """
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                tmp = deepcopy(transition)
                tmp["observations"]["state"] = np.concatenate(
                    (
                        tmp["observations"]["state"][:, :4],
                        tmp["observations"]["state"][:, 6][None, ...],
                        tmp["observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                tmp["next_observations"]["state"] = np.concatenate(
                    (
                        tmp["next_observations"]["state"][:, :4],
                        tmp["next_observations"]["state"][:, 6][None, ...],
                        tmp["next_observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                data_store.insert(tmp)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store
