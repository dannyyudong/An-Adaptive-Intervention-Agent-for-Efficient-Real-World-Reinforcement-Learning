import unittest

import numpy as np

from serl_launcher.aia.options import OptionID, OptionResult, TerminationReason
from serl_launcher.aia.transition import (
    SchedulerRewardConfig,
    SchedulerTransitionBuilder,
)


def _result(
    option_id=OptionID.TRAJECTORY_CORRECTION,
    duration=20,
    task_return=0.5,
    terminated=False,
    truncated=False,
):
    return OptionResult(
        option_id=option_id,
        start_step=10,
        duration=duration,
        discounted_return=task_return,
        termination_reason=TerminationReason.TARGET_REACHED,
        confidence=0.8,
        available_at_start=True,
        episode_terminated=terminated,
        episode_truncated=truncated,
    )


class SchedulerTransitionBuilderTest(unittest.TestCase):
    def setUp(self):
        self.reward_config = SchedulerRewardConfig(
            gamma=0.9,
            trajectory_cost=0.02,
            code_policy_cost=0.05,
            duration_cost=0.1,
            max_duration=100,
        )
        self.builder = SchedulerTransitionBuilder(6, self.reward_config)

    def test_reward_and_smdp_discount(self):
        result = _result()
        reward = self.reward_config.reward(result)
        self.assertAlmostEqual(reward, 0.5 - 0.02 - 0.1 * 0.2)

        pending = self.builder.begin(
            np.arange(6, dtype=np.float32),
            result.option_id,
            np.array([True, True, False]),
            behavior_prob=0.75,
            start_step=10,
        )
        transition = self.builder.finish(
            pending,
            result,
            np.arange(6, dtype=np.float32) + 1.0,
            np.array([True, False, False]),
            scheduler_reward=reward,
        )

        self.assertEqual(transition["durations"], 20)
        self.assertAlmostEqual(float(transition["discounts"]), 0.9**20)
        self.assertFalse(transition["dones"])
        self.assertEqual(transition["rl_policy_version"], 0)
        self.assertNotIn("penalized_reset", transition)

    def test_terminal_transition_allows_empty_next_mask(self):
        result = _result(terminated=True)
        pending = self.builder.begin(
            np.zeros(6, dtype=np.float32),
            result.option_id,
            np.array([True, True, False]),
            behavior_prob=1.0,
            start_step=10,
        )
        transition = self.builder.finish(
            pending,
            result,
            np.ones(6, dtype=np.float32),
            np.zeros(3, dtype=bool),
            scheduler_reward=0.5,
        )
        self.assertTrue(transition["dones"])
        self.assertTrue(transition["terminated"])

    def test_rejects_zero_duration(self):
        result = _result(duration=0)
        pending = self.builder.begin(
            np.zeros(6, dtype=np.float32),
            result.option_id,
            np.array([True, True, False]),
            behavior_prob=1.0,
            start_step=10,
        )
        with self.assertRaisesRegex(ValueError, "zero duration"):
            self.builder.finish(
                pending,
                result,
                np.ones(6, dtype=np.float32),
                np.array([True, False, False]),
                scheduler_reward=0.0,
            )

    def test_records_task_specific_state_schema(self):
        builder = SchedulerTransitionBuilder(
            6,
            self.reward_config,
            state_schema_version="circle-policy-change-v1",
        )
        result = _result(option_id=OptionID.RL, duration=5)
        pending = builder.begin(
            np.zeros(6, dtype=np.float32),
            result.option_id,
            np.array([True, True, False]),
            behavior_prob=1.0,
            start_step=10,
        )
        transition = builder.finish(
            pending,
            result,
            np.ones(6, dtype=np.float32),
            np.array([True, False, False]),
            scheduler_reward=0.0,
        )

        self.assertEqual(transition["state_schema_version"], "circle-policy-change-v1")


if __name__ == "__main__":
    unittest.main()
