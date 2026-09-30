import unittest
import tempfile
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from flax import config as flax_config
from flax.training import checkpoints

from serl_launcher.aia.dqn import (
    SchedulerDQNAgent,
    SchedulerDQNConfig,
    masked_double_dqn_targets,
)


class SchedulerDQNTest(unittest.TestCase):
    def setUp(self):
        self.config = SchedulerDQNConfig(
            hidden_dims=(16,),
            learning_rate=1e-3,
            batch_size=2,
            warmup_transitions=2,
            updates_per_transition=1,
            max_updates_per_loop=2,
            target_update_tau=0.1,
            gradient_clip=1.0,
            epsilon_start=0.5,
            epsilon_end=0.1,
            epsilon_decay_steps=10,
        )
        self.agent = SchedulerDQNAgent.create(
            jax.random.PRNGKey(0),
            state_dim=6,
            num_options=3,
            config=self.config,
        )

    def test_smdp_double_dqn_target_uses_discount_and_mask(self):
        targets, next_actions = masked_double_dqn_targets(
            rewards=jnp.array([1.0, 1.0, 2.0]),
            masks=jnp.array([1.0, 1.0, 0.0]),
            discounts=jnp.array([0.9, 0.5, 0.9]),
            online_next_q=jnp.array(
                [[1.0, 5.0, 3.0], [4.0, 3.0, 2.0], [9.0, 8.0, 7.0]]
            ),
            target_next_q=jnp.array(
                [[10.0, 20.0, 30.0], [40.0, 30.0, 20.0], [90.0, 80.0, 70.0]]
            ),
            next_available_actions=jnp.array(
                [[True, False, True], [False, True, True], [False, False, False]]
            ),
        )
        np.testing.assert_array_equal(np.asarray(next_actions), [2, 1, 0])
        np.testing.assert_allclose(
            np.asarray(targets),
            np.array([1.0 + 0.9 * 30.0, 1.0 + 0.5 * 30.0, 2.0]),
        )

    def test_masked_epsilon_greedy_never_returns_invalid_option(self):
        mask = np.array([False, True, False])
        for seed in range(10):
            action, probability, q_values = self.agent.sample_action(
                np.zeros(6, dtype=np.float32),
                mask,
                seed=jax.random.PRNGKey(seed),
                epsilon=1.0,
            )
            self.assertEqual(action, 1)
            self.assertEqual(probability, 1.0)
            self.assertEqual(q_values.shape, (3,))

    def test_weighted_exploration_favors_trajectory_and_reports_probability(self):
        config = replace(
            self.config,
            exploration_weights=(1.0, 2.0, 0.25),
        )
        agent = SchedulerDQNAgent.create(
            jax.random.PRNGKey(11),
            state_dim=6,
            num_options=3,
            config=config,
        )
        expected_probabilities = np.asarray((1.0, 2.0, 0.25)) / 3.25
        counts = np.zeros((3,), dtype=np.int32)

        for seed in range(128):
            action, probability, _ = agent.sample_action(
                np.zeros(6, dtype=np.float32),
                np.ones(3, dtype=bool),
                seed=jax.random.PRNGKey(seed),
                epsilon=1.0,
            )
            counts[action] += 1
            self.assertAlmostEqual(
                probability,
                expected_probabilities[action],
                places=6,
            )

        self.assertGreater(counts[1], counts[0])
        self.assertGreater(counts[0], counts[2])

    def test_weighted_exploration_renormalizes_over_available_options(self):
        config = replace(
            self.config,
            exploration_weights=(1.0, 2.0, 0.25),
        )
        agent = SchedulerDQNAgent.create(
            jax.random.PRNGKey(12),
            state_dim=6,
            num_options=3,
            config=config,
        )
        action_mask = np.array([True, True, False])

        for seed in range(32):
            action, probability, _ = agent.sample_action(
                np.zeros(6, dtype=np.float32),
                action_mask,
                seed=jax.random.PRNGKey(seed),
                epsilon=1.0,
            )
            self.assertIn(action, (0, 1))
            self.assertAlmostEqual(
                probability,
                (1.0 / 3.0, 2.0 / 3.0)[action],
                places=6,
            )

    def test_weighted_epsilon_greedy_probability_includes_greedy_branch(self):
        config = replace(
            self.config,
            exploration_weights=(1.0, 2.0, 0.25),
        )
        agent = SchedulerDQNAgent.create(
            jax.random.PRNGKey(13),
            state_dim=6,
            num_options=3,
            config=config,
        )
        observation = np.zeros(6, dtype=np.float32)
        q_values = np.asarray(agent.q_values(jnp.asarray(observation)))
        greedy_action = int(np.argmax(q_values))
        expected_exploration = np.asarray((1.0, 2.0, 0.25)) / 3.25

        for seed in range(32):
            action, probability, _ = agent.sample_action(
                observation,
                np.ones(3, dtype=bool),
                seed=jax.random.PRNGKey(seed),
                epsilon=0.3,
            )
            expected = 0.3 * expected_exploration[action]
            if action == greedy_action:
                expected += 0.7
            self.assertAlmostEqual(probability, expected, places=6)

    def test_exploration_weights_must_match_option_count(self):
        config = replace(self.config, exploration_weights=(1.0, 2.0))

        with self.assertRaisesRegex(ValueError, "one weight per Option"):
            SchedulerDQNAgent.create(
                jax.random.PRNGKey(14),
                state_dim=6,
                num_options=3,
                config=config,
            )

    def test_action_sampling_is_available_before_learning_warmup(self):
        config = SchedulerDQNConfig(
            hidden_dims=(16,),
            batch_size=2,
            warmup_transitions=64,
            epsilon_start=0.3,
            epsilon_end=0.05,
            epsilon_decay_steps=100,
        )
        agent = SchedulerDQNAgent.create(
            jax.random.PRNGKey(1),
            state_dim=6,
            num_options=3,
            config=config,
        )
        action, probability, q_values = agent.sample_action(
            np.zeros(6, dtype=np.float32),
            np.array([True, True, True]),
            seed=jax.random.PRNGKey(2),
            epsilon=config.epsilon(0),
        )

        self.assertIn(action, (0, 1, 2))
        self.assertGreater(probability, 0.0)
        self.assertEqual(q_values.shape, (3,))
        self.assertEqual(int(agent.state.step), 0)

    def test_update_is_finite_and_changes_only_scheduler_state(self):
        batch = {
            "observations": jnp.zeros((2, 6), dtype=jnp.float32),
            "actions": jnp.array([0, 2], dtype=jnp.int32),
            "rewards": jnp.array([0.5, -0.1], dtype=jnp.float32),
            "next_observations": jnp.ones((2, 6), dtype=jnp.float32),
            "durations": jnp.array([5, 20], dtype=jnp.int32),
            "discounts": jnp.array([0.9**5, 0.9**20], dtype=jnp.float32),
            "masks": jnp.array([1.0, 0.0], dtype=jnp.float32),
            "next_available_actions": jnp.array(
                [[True, True, False], [False, False, False]]
            ),
        }
        old_params = self.agent.state.params
        new_agent, info = self.agent.update(batch)

        self.assertTrue(np.isfinite(float(info["loss"])))
        self.assertTrue(np.isfinite(float(info["grad_norm"])))
        self.assertEqual(int(new_agent.state.step), 1)
        changed = jax.tree_util.tree_leaves(
            jax.tree.map(
                lambda x, y: jnp.any(x != y), old_params, new_agent.state.params
            )
        )
        self.assertTrue(any(bool(np.asarray(value)) for value in changed))

    def test_epsilon_schedule(self):
        self.assertAlmostEqual(self.config.epsilon(0), 0.5)
        self.assertAlmostEqual(self.config.epsilon(5), 0.3)
        self.assertAlmostEqual(self.config.epsilon(100), 0.1)

    def test_scheduler_checkpoint_round_trip_preserves_q_values(self):
        observation = jnp.arange(6, dtype=jnp.float32)
        expected_q = np.asarray(self.agent.q_values(observation))
        use_orbax = flax_config.flax_use_orbax_checkpointing
        flax_config.update("flax_use_orbax_checkpointing", False)
        try:
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                checkpoints.save_checkpoint(
                    checkpoint_dir,
                    self.agent.state,
                    step=7,
                    prefix="scheduler_",
                )
                restored_state = checkpoints.restore_checkpoint(
                    checkpoint_dir,
                    self.agent.state,
                    prefix="scheduler_",
                )
        finally:
            flax_config.update("flax_use_orbax_checkpointing", use_orbax)
        restored_agent = self.agent.replace(state=restored_state)
        np.testing.assert_allclose(
            np.asarray(restored_agent.q_values(observation)), expected_q
        )


if __name__ == "__main__":
    unittest.main()
