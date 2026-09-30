import unittest

import numpy as np

from serl_launcher.data.data_store import SchedulerReplayBufferDataStore


def _transition(value: float, state_dim: int = 6):
    done = value < 0.0
    return {
        "observations": np.full(state_dim, value, dtype=np.float32),
        "actions": np.int32(1),
        "rewards": np.float32(value),
        "next_observations": np.full(state_dim, value + 1.0, dtype=np.float32),
        "durations": np.int32(5),
        "discounts": np.float32(0.9**5),
        "masks": np.float32(1.0 - float(done)),
        "dones": np.bool_(done),
        "terminated": np.bool_(done),
        "truncated": np.bool_(False),
        "termination_reason": np.int32(0),
        "behavior_prob": np.float32(1.0),
        "available_actions": np.array([True, True, False]),
        "next_available_actions": np.array([True, False, False]),
        "start_step": np.int64(10),
        "rl_policy_version": np.int64(7),
        # Local pickle/debug metadata is intentionally ignored by replay arrays.
        "option_name": "TRAJECTORY_CORRECTION",
        "schema_version": "test",
    }


class SchedulerReplayBufferDataStoreTest(unittest.TestCase):
    def test_capacity_overwrite_and_batch_schema(self):
        replay = SchedulerReplayBufferDataStore(
            state_dim=6,
            num_options=3,
            capacity=2,
        )
        replay.insert(_transition(1.0))
        replay.insert(_transition(2.0))
        replay.insert(_transition(3.0))

        self.assertEqual(len(replay), 2)
        self.assertEqual(replay.latest_data_id(), 3)
        np.testing.assert_array_equal(
            np.sort(replay.dataset_dict["rewards"][:2]),
            np.array([2.0, 3.0], dtype=np.float32),
        )

        batch = replay.sample(batch_size=2, indx=np.array([0, 1]))
        self.assertEqual(batch["observations"].shape, (2, 6))
        self.assertEqual(batch["next_observations"].shape, (2, 6))
        self.assertEqual(batch["available_actions"].shape, (2, 3))
        self.assertEqual(batch["actions"].dtype, np.int64)
        self.assertEqual(batch["durations"].dtype, np.int32)
        self.assertEqual(batch["discounts"].dtype, np.float32)
        self.assertEqual(batch["dones"].dtype, np.bool_)
        self.assertEqual(batch["rl_policy_version"].dtype, np.int64)
        self.assertNotIn("option_name", batch)
        self.assertNotIn("schema_version", batch)

    def test_agentlace_incremental_data_keeps_full_transition(self):
        replay = SchedulerReplayBufferDataStore(
            state_dim=6,
            num_options=3,
            capacity=4,
        )
        transition = _transition(-1.0)
        replay.insert(transition)

        latest = replay.get_latest_data(0)

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["option_name"], transition["option_name"])
        self.assertTrue(latest[0]["dones"])

    def test_recent_sampling_follows_ring_buffer_order(self):
        replay = SchedulerReplayBufferDataStore(
            state_dim=6,
            num_options=3,
            capacity=5,
        )
        replay.seed(0)
        for value in range(7):
            replay.insert(_transition(float(value)))

        batch = replay.sample(
            batch_size=64,
            recent_fraction=1.0,
            recent_window=2,
        )

        self.assertTrue(set(np.asarray(batch["rewards"])).issubset({5.0, 6.0}))


if __name__ == "__main__":
    unittest.main()
