from types import SimpleNamespace
import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from serl_launcher.common.common import ModuleDict
from serl_launcher.networks.actor_critic_nets import Policy
from serl_launcher.aia.options import OptionID, OptionResult, TerminationReason
from serl_launcher.aia.state import (
    OPTION_HISTORY_ITEM_DIM,
    FrozenObservationEncoder,
    OptionHistory,
    OptionHistoryConfig,
    OptionMotionAccumulator,
    OptionMotionSummary,
    SchedulerStateBuilder,
)


def _fake_apply_fn(
    variables,
    observation,
    *,
    name,
    train,
    return_features,
):
    assert name == "actor"
    assert train is False
    assert return_features is True
    encoder = variables["params"]["modules_actor"]["encoder"]
    state = jnp.asarray(observation["state"], dtype=jnp.float32)
    return state * encoder["scale"] + encoder["bias"]


def _fake_agent(scale, bias, step=0):
    params = {
        "modules_actor": {
            "encoder": {
                "scale": jnp.asarray(scale, dtype=jnp.float32),
                "bias": jnp.asarray(bias, dtype=jnp.float32),
            },
            "policy_head": {"unused": jnp.ones((2,), dtype=jnp.float32)},
        },
        "modules_critic": {"unused": jnp.ones((2,), dtype=jnp.float32)},
    }
    state = SimpleNamespace(params=params, apply_fn=_fake_apply_fn, step=step)
    return SimpleNamespace(state=state)


class _TinyEncoder(nn.Module):
    @nn.compact
    def __call__(self, observation, *, train=False, stop_gradient=False):
        del train
        features = nn.Dense(3, name="projection")(observation["state"])
        if stop_gradient:
            features = jax.lax.stop_gradient(features)
        return features


class _TinyNetwork(nn.Module):
    @nn.compact
    def __call__(self, features, *, train=False):
        del train
        return nn.Dense(4)(features)


class FrozenObservationEncoderTest(unittest.TestCase):
    def test_real_policy_feature_path_works_with_encoder_only_params(self):
        observation = {"state": jnp.array([1.0, 2.0], dtype=jnp.float32)}
        policy = Policy(
            encoder=_TinyEncoder(),
            network=_TinyNetwork(),
            action_dim=2,
            name="actor",
        )
        model = ModuleDict({"actor": policy})
        params = model.init(jax.random.PRNGKey(0), actor=[observation])["params"]
        agent = SimpleNamespace(
            state=SimpleNamespace(
                params=params,
                apply_fn=model.apply,
                step=5,
            )
        )

        encoder = FrozenObservationEncoder.from_agent(
            agent,
            observation,
            expected_feature_dim=3,
        )

        self.assertEqual(encoder.encode(observation).shape, (3,))
        self.assertEqual(encoder.source_step, 5)

    def test_snapshot_is_independent_from_later_agent_params(self):
        observation = {"state": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        agent = _fake_agent([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], step=17)

        encoder = FrozenObservationEncoder.from_agent(
            agent,
            observation,
            expected_feature_dim=3,
        )
        before = encoder.encode(observation)

        agent.state.params["modules_actor"]["encoder"]["scale"] = jnp.full((3,), 9.0)
        after = encoder.encode(observation)

        np.testing.assert_array_equal(before, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(after, before)
        self.assertEqual(encoder.source_step, 17)

    def test_builder_exposes_fixed_dimension_float32_state(self):
        observation = {"state": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        encoder = FrozenObservationEncoder.from_agent(
            _fake_agent([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]),
            observation,
        )
        history_config = OptionHistoryConfig(length=2)
        history = OptionHistory(history_config)
        builder = SchedulerStateBuilder(encoder, history_config)

        state = builder.build(observation, history)

        expected_dim = 3 + 2 * (OPTION_HISTORY_ITEM_DIM + 1)
        self.assertEqual(builder.state_dim, expected_dim)
        self.assertEqual(state.shape, (expected_dim,))
        self.assertEqual(state.dtype, np.float32)
        np.testing.assert_array_equal(state[:3], np.array([3.0, 5.0, 7.0]))
        np.testing.assert_array_equal(state[-2:], np.zeros(2, dtype=np.float32))

    def test_builder_appends_validated_task_local_evidence(self):
        observation = {"state": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        encoder = FrozenObservationEncoder.from_agent(
            _fake_agent(np.ones(3), np.zeros(3)), observation
        )
        history_config = OptionHistoryConfig(length=1)
        builder = SchedulerStateBuilder(
            encoder,
            history_config,
            extra_feature_dim=2,
            extra_feature_fn=lambda _obs: np.array([0.25, 0.75]),
            schema_version="task-local-v1",
        )

        state = builder.build(observation, OptionHistory(history_config))

        self.assertEqual(builder.state_dim, 3 + OPTION_HISTORY_ITEM_DIM + 1 + 2)
        self.assertEqual(builder.schema_version, "task-local-v1")
        np.testing.assert_array_equal(state[-2:], [0.25, 0.75])

    def test_builder_rejects_mismatched_extra_feature_contract(self):
        observation = {"state": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        encoder = FrozenObservationEncoder.from_agent(
            _fake_agent(np.ones(3), np.zeros(3)), observation
        )
        with self.assertRaisesRegex(ValueError, "extra_feature_fn"):
            SchedulerStateBuilder(
                encoder,
                OptionHistoryConfig(length=1),
                extra_feature_dim=2,
            )

    def test_tube_scheduler_state_dimension_is_648(self):
        observation = {"state": np.zeros(576, dtype=np.float32)}
        encoder = FrozenObservationEncoder.from_agent(
            _fake_agent(np.ones(576), np.zeros(576)),
            observation,
            expected_feature_dim=576,
        )
        history_config = OptionHistoryConfig(length=4)
        builder = SchedulerStateBuilder(encoder, history_config)

        state = builder.build(observation, OptionHistory(history_config))

        self.assertEqual(builder.state_dim, 648)
        self.assertEqual(state.shape, (648,))

    def test_rejects_non_finite_features(self):
        observation = {"state": np.array([1.0, np.nan], dtype=np.float32)}
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            FrozenObservationEncoder.from_agent(
                _fake_agent([1.0, 1.0], [0.0, 0.0]),
                observation,
            )


def _option_result(option_id=OptionID.RL, duration=10, reward=0.25):
    return OptionResult(
        option_id=option_id,
        start_step=3,
        duration=duration,
        discounted_return=reward,
        termination_reason=TerminationReason.HORIZON_REACHED,
        confidence=0.7,
        available_at_start=True,
        episode_terminated=False,
        episode_truncated=False,
    )


class OptionHistoryTest(unittest.TestCase):
    def test_motion_accumulator_distinguishes_net_delta_and_path_length(self):
        start = {
            "state": np.array(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                dtype=np.float32,
            )
        }
        middle = {
            "state": np.array(
                [[3.0, 4.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]],
                dtype=np.float32,
            )
        }
        end = {
            "state": np.array(
                [[0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]],
                dtype=np.float32,
            )
        }
        accumulator = OptionMotionAccumulator(start)
        accumulator.observe(middle)
        accumulator.observe(end)

        motion = accumulator.finish()

        np.testing.assert_allclose(motion.delta_xyz, np.zeros(3), atol=1e-7)
        self.assertAlmostEqual(motion.path_length_m, 0.02, places=6)
        np.testing.assert_allclose(motion.delta_force, [0.0, 0.0, 10.0])
        self.assertAlmostEqual(motion.mean_force_magnitude_n, 7.5, places=6)
        self.assertAlmostEqual(motion.max_force_magnitude_n, 10.0, places=6)
        self.assertGreater(np.linalg.norm(motion.delta_rotvec), 0.0)

    def test_motion_accumulator_supports_learned_gripper_state(self):
        start = {
            "state": np.array(
                [[0.1, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                dtype=np.float32,
            )
        }
        end = {
            "state": np.array(
                [[0.9, 1.0, 4.0, 6.0, 3.0, 0.03, 0.04, 0.0, 0.0, 0.0, 0.1]],
                dtype=np.float32,
            )
        }

        accumulator = OptionMotionAccumulator(start)
        accumulator.observe(end)
        motion = accumulator.finish()

        np.testing.assert_allclose(motion.delta_xyz, [0.03, 0.04, 0.0])
        self.assertAlmostEqual(motion.path_length_m, 0.05, places=6)
        np.testing.assert_allclose(motion.delta_force, [3.0, 4.0, 0.0])
        self.assertGreater(np.linalg.norm(motion.delta_rotvec), 0.0)

    def test_history_is_left_padded_fifo_without_reason_or_confidence(self):
        config = OptionHistoryConfig(
            length=2,
            max_duration=100,
            reward_scale=1.0,
            position_scale_m=1.0,
            rotation_scale_rad=1.0,
            path_length_scale_m=1.0,
            force_scale_n=10.0,
        )
        history = OptionHistory(config)
        zero_motion = OptionMotionSummary(
            delta_xyz=np.zeros(3),
            delta_rotvec=np.zeros(3),
            path_length_m=0.0,
            delta_force=np.zeros(3),
            mean_force_magnitude_n=0.0,
            max_force_magnitude_n=0.0,
        )

        history.append(_option_result(OptionID.RL, 10), 0.1, zero_motion)
        encoded, mask = history.encode()
        self.assertEqual(encoded.shape, (2, OPTION_HISTORY_ITEM_DIM))
        np.testing.assert_array_equal(mask, [0.0, 1.0])
        np.testing.assert_array_equal(encoded[-1, :3], [1.0, 0.0, 0.0])

        history.append(
            _option_result(OptionID.TRAJECTORY_CORRECTION, 20),
            -0.2,
            zero_motion,
        )
        history.append(_option_result(OptionID.CODE_POLICY, 30), 0.8, zero_motion)
        encoded, mask = history.encode()

        np.testing.assert_array_equal(mask, [1.0, 1.0])
        np.testing.assert_array_equal(encoded[0, :3], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(encoded[1, :3], [0.0, 0.0, 1.0])
        self.assertAlmostEqual(float(encoded[0, 3]), 0.2)
        self.assertAlmostEqual(float(encoded[1, 3]), 0.3)
        self.assertAlmostEqual(float(encoded[0, 4]), -0.2)
        self.assertAlmostEqual(float(encoded[1, 4]), 0.8)


if __name__ == "__main__":
    unittest.main()
