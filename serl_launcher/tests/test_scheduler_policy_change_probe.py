import copy
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from serl_launcher.aia.policy_change_probe import (
    POLICY_CHANGE_PROBE_OUTCOME_FEATURE_DIM,
    PolicyChangeProbeConfig,
    PolicyChangeProbeController,
    PolicySignature,
    RegionalAnchorSet,
    policy_signature_symmetric_kl,
)


def _signature(mean=0.0, std=1.0, logits=(0.0, 0.0, 0.0)):
    return PolicySignature(
        continuous_mean=np.full((2, 3), mean, dtype=np.float32),
        continuous_std=np.full((2, 3), std, dtype=np.float32),
        discrete_logits=np.tile(np.asarray(logits, dtype=np.float32), (2, 1)),
    )


def _config(**overrides):
    values = {
        "region_names": ("approach", "transport", "place"),
        "drift_threshold": 0.05,
        "max_age_steps": 100,
        "budget_window_steps": 1000,
        "budget_steps": 10,
        "rl_option_horizon": 5,
        "region_max_horizons": (15, 20, 10),
        "evidence_history_length": 2,
    }
    values.update(overrides)
    return PolicyChangeProbeConfig(**values)


class NoDriftProbeTest(unittest.TestCase):
    def test_actor_refresh_skips_network_inference(self):
        source = Path(__file__).resolve().parents[2] / "examples/train_aia.py"
        tree = ast.parse(source.read_text())
        refresh = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "refresh_policy_change_probe_signatures"
        )
        # Isolate the real nested refresh entrypoint without importing hardware.
        refresh.body = [
            node for node in refresh.body if not isinstance(node, ast.Nonlocal)
        ]
        controller = SimpleNamespace(
            config=SimpleNamespace(use_policy_drift=False), current_policy_version=0
        )
        namespace = {
            "policy_change_probe_controller": controller,
            "rl_policy_version": 12,
        }
        exec(
            compile(ast.Module(body=[refresh], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        self.assertFalse(
            namespace["refresh_policy_change_probe_signatures"](force=True)
        )
        self.assertEqual(controller.current_policy_version, 12)

    def test_probe_without_signatures_and_restore(self):
        controller = PolicyChangeProbeController(
            _config(use_policy_drift=False), anchor_fingerprint="test"
        )
        with patch(
            "serl_launcher.aia.policy_change_probe.policy_signature_symmetric_kl",
            side_effect=AssertionError("KL must not run"),
        ):
            decision = controller.decide(
                base_is_rl=False, rl_available=True, region=0, step=0
            )
            self.assertEqual(decision.reason, "unseen")
            controller.reserve_probe(decision, start_step=0)
            controller.record_autonomous_execution(
                region=0,
                signature=None,
                start_step=0,
                duration=5,
                task_return=0.0,
                termination_reason="horizon_reached",
                success=False,
                episode_terminated=False,
                episode_truncated=False,
                probe_decision=decision,
            )
            self.assertFalse(
                controller.decide(
                    base_is_rl=False, rl_available=True, region=0, step=6
                ).override
            )
            self.assertEqual(
                controller.decide(
                    base_is_rl=False, rl_available=True, region=0, step=105
                ).reason,
                "age",
            )
            features = controller.state_features(region=0, step=6)
            np.testing.assert_array_equal(features[5:8], np.zeros(3))
            restored = PolicyChangeProbeController(
                _config(use_policy_drift=False), anchor_fingerprint="test"
            )
            restored.restore_state(controller.state_dict())
            np.testing.assert_array_equal(
                restored.state_features(region=0, step=6), features
            )

    def test_legacy_state_load_discards_signatures(self):
        legacy = PolicyChangeProbeController(_config(), anchor_fingerprint="test")
        legacy.set_current_signatures([_signature()] * 3, policy_version=1)
        legacy.record_autonomous_execution(
            region=0,
            signature=_signature(),
            start_step=0,
            duration=5,
            task_return=0.0,
            termination_reason="horizon_reached",
            success=False,
            episode_terminated=False,
            episode_truncated=False,
        )
        state = legacy.state_dict()
        del state["config"]["use_policy_drift"]
        controller = PolicyChangeProbeController(
            _config(use_policy_drift=False), anchor_fingerprint="test"
        )
        controller.restore_state(state)
        self.assertIsNone(controller.current_signature(0))
        self.assertEqual(
            controller.decide(
                base_is_rl=False, rl_available=True, region=0, step=6
            ).reason,
            "fresh_evidence",
        )


class PolicySignatureTest(unittest.TestCase):
    def test_identical_signature_has_zero_symmetric_kl(self):
        drift = policy_signature_symmetric_kl(_signature(), _signature())

        self.assertAlmostEqual(drift.continuous_skl, 0.0, places=7)
        self.assertAlmostEqual(drift.discrete_skl, 0.0, places=7)
        self.assertAlmostEqual(drift.combined_skl, 0.0, places=7)

    def test_gaussian_and_discrete_changes_are_detected(self):
        drift = policy_signature_symmetric_kl(
            _signature(mean=1.0, std=2.0, logits=(4.0, 0.0, -1.0)),
            _signature(),
        )

        self.assertGreater(drift.continuous_skl, 0.0)
        self.assertGreater(drift.discrete_skl, 0.0)
        self.assertGreater(drift.combined_skl, 0.0)


class RegionalAnchorSetTest(unittest.TestCase):
    def test_maps_progress_to_fixed_regions(self):
        anchor_set = RegionalAnchorSet(
            region_names=("a", "b", "c"),
            region_start_indices=(0, 4, 8),
            anchor_observations=(({},), ({},), ({},)),
            trajectory_length=10,
            fingerprint="abc",
        )

        self.assertEqual(anchor_set.region_for_progress_index(-2), 0)
        self.assertEqual(anchor_set.region_for_progress_index(3), 0)
        self.assertEqual(anchor_set.region_for_progress_index(4), 1)
        self.assertEqual(anchor_set.region_for_progress_index(9), 2)
        self.assertEqual(anchor_set.region_for_progress_index(99), 2)
        self.assertEqual(anchor_set.region_stop_indices, (4, 8, 10))
        self.assertEqual(anchor_set.region_transition_counts(), (4, 4, 2))
        self.assertEqual(anchor_set.region_max_horizons(5), (5, 5, 5))
        self.assertEqual(anchor_set.remaining_region_steps(4), 4)


class PolicyChangeProbeControllerTest(unittest.TestCase):
    def setUp(self):
        self.controller = PolicyChangeProbeController(
            _config(), anchor_fingerprint="demo-v1"
        )
        self.controller.set_current_signatures(
            (_signature(), _signature(), _signature()), policy_version=0
        )

    def test_unseen_requests_only_override_non_rl_scheduler_choice(self):
        base_rl = self.controller.decide(
            base_is_rl=True, rl_available=True, region=0, step=0
        )
        recovery = self.controller.decide(
            base_is_rl=False, rl_available=True, region=0, step=0
        )

        self.assertFalse(base_rl.override)
        self.assertEqual(base_rl.reason, "base_rl")
        self.assertTrue(recovery.override)
        self.assertEqual(recovery.trigger_reasons, ("unseen",))

    def test_normal_rl_refreshes_evidence_without_spending_probe_budget(self):
        self.controller.record_autonomous_execution(
            region=0,
            signature=self.controller.current_signature(0),
            start_step=0,
            duration=5,
            task_return=0.0,
            termination_reason="horizon_reached",
            success=False,
            episode_terminated=False,
            episode_truncated=False,
        )

        decision = self.controller.decide(
            base_is_rl=False, rl_available=True, region=0, step=10
        )
        features = self.controller.state_features(region=0, step=10)

        self.assertFalse(decision.override)
        self.assertEqual(decision.reason, "fresh_evidence")
        self.assertEqual(decision.remaining_budget_steps, 10)
        self.assertEqual(features.shape, (self.controller.state_feature_dim,))
        self.assertEqual(
            features.shape[0],
            3 + 6 + 2 * POLICY_CHANGE_PROBE_OUTCOME_FEATURE_DIM,
        )
        self.assertEqual(features[3], 1.0)  # seen flag after region one-hot
        self.assertEqual(features[-POLICY_CHANGE_PROBE_OUTCOME_FEATURE_DIM], 1.0)

    def test_policy_change_triggers_and_actual_probe_duration_is_charged(self):
        self.controller.record_autonomous_execution(
            region=0,
            signature=self.controller.current_signature(0),
            start_step=0,
            duration=5,
            task_return=0.0,
            termination_reason="horizon_reached",
            success=False,
            episode_terminated=False,
            episode_truncated=False,
        )
        changed = _signature(mean=1.0)
        self.controller.set_current_signatures(
            (changed, _signature(), _signature()), policy_version=50
        )
        decision = self.controller.decide(
            base_is_rl=False, rl_available=True, region=0, step=50
        )

        self.assertTrue(decision.override)
        self.assertIn("policy_change", decision.trigger_reasons)
        self.controller.reserve_probe(decision, start_step=50)
        self.controller.record_autonomous_execution(
            region=0,
            signature=self.controller.current_signature(0),
            start_step=50,
            duration=3,
            task_return=0.25,
            termination_reason="episode_terminated",
            success=True,
            episode_terminated=True,
            episode_truncated=False,
            probe_decision=decision,
        )
        after = self.controller.decide(
            base_is_rl=False, rl_available=True, region=1, step=53
        )

        self.assertEqual(after.remaining_budget_steps, 7)
        self.assertAlmostEqual(self.controller.drift(0).combined_skl, 0.0)

    def test_budget_and_window_boundary_can_deny_a_request(self):
        near_boundary = self.controller.decide(
            base_is_rl=False, rl_available=True, region=0, step=998
        )
        self.assertFalse(near_boundary.override)
        self.assertEqual(near_boundary.reason, "window_boundary")

        first = self.controller.decide(
            base_is_rl=False, rl_available=True, region=0, step=1000
        )
        self.controller.reserve_probe(first, start_step=1000)
        self.controller.record_autonomous_execution(
            region=0,
            signature=self.controller.current_signature(0),
            start_step=1000,
            duration=5,
            task_return=0.0,
            termination_reason="horizon_reached",
            success=False,
            episode_terminated=False,
            episode_truncated=False,
            probe_decision=first,
        )
        second = self.controller.decide(
            base_is_rl=False, rl_available=True, region=1, step=1005
        )
        self.controller.reserve_probe(second, start_step=1005)
        self.controller.record_autonomous_execution(
            region=1,
            signature=self.controller.current_signature(1),
            start_step=1005,
            duration=5,
            task_return=0.0,
            termination_reason="horizon_reached",
            success=False,
            episode_terminated=False,
            episode_truncated=False,
            probe_decision=second,
        )
        denied = self.controller.decide(
            base_is_rl=False, rl_available=True, region=2, step=1010
        )

        self.assertFalse(denied.override)
        self.assertTrue(denied.requested)
        self.assertEqual(denied.reason, "budget_exhausted")

    def test_state_round_trip_preserves_evidence_and_budget(self):
        decision = self.controller.decide(
            base_is_rl=False, rl_available=True, region=0, step=0
        )
        self.controller.reserve_probe(decision, start_step=0)
        self.controller.record_autonomous_execution(
            region=0,
            signature=self.controller.current_signature(0),
            start_step=0,
            duration=2,
            task_return=-0.1,
            termination_reason="human_intervention",
            success=False,
            episode_terminated=False,
            episode_truncated=False,
            probe_decision=decision,
        )
        saved = copy.deepcopy(self.controller.state_dict())
        restored = PolicyChangeProbeController(_config(), anchor_fingerprint="demo-v1")
        restored.restore_state(saved)
        restored.set_current_signatures(
            (_signature(), _signature(), _signature()), policy_version=10
        )

        decision_after_restore = restored.decide(
            base_is_rl=False, rl_available=True, region=1, step=10
        )
        self.assertEqual(decision_after_restore.remaining_budget_steps, 8)
        np.testing.assert_array_equal(
            restored.state_features(region=0, step=10),
            self.controller.state_features(region=0, step=10),
        )

    def test_inflight_reservation_survives_restart_as_conservative_charge(self):
        decision = self.controller.decide(
            base_is_rl=False, rl_available=True, region=0, step=0
        )
        self.controller.reserve_probe(decision, start_step=0)
        saved = copy.deepcopy(self.controller.state_dict())
        restored = PolicyChangeProbeController(_config(), anchor_fingerprint="demo-v1")
        restored.restore_state(saved)
        restored.set_current_signatures(
            (_signature(), _signature(), _signature()), policy_version=10
        )

        after_restart = restored.decide(
            base_is_rl=False, rl_available=True, region=1, step=10
        )

        self.assertEqual(after_restart.remaining_budget_steps, 5)

    def test_zero_step_cancellation_releases_probe_reservation(self):
        decision = self.controller.decide(
            base_is_rl=False, rl_available=True, region=0, step=0
        )
        self.controller.reserve_probe(decision, start_step=0)

        self.controller.cancel_probe_reservation(decision, start_step=0)
        after_cancel = self.controller.decide(
            base_is_rl=False, rl_available=True, region=1, step=0
        )

        self.assertTrue(after_cancel.override)
        self.assertEqual(after_cancel.remaining_budget_steps, 10)

    def test_nonfinite_outcome_return_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "task_return must be finite"):
            self.controller.record_autonomous_execution(
                region=0,
                signature=self.controller.current_signature(0),
                start_step=0,
                duration=1,
                task_return=np.nan,
                termination_reason="horizon_reached",
                success=False,
                episode_terminated=False,
                episode_truncated=False,
            )

    def test_probe_session_owns_multiple_five_step_options(self):
        controller = PolicyChangeProbeController(
            _config(
                budget_steps=20,
                initial_horizon_steps=10,
                required_passes=1,
            ),
            anchor_fingerprint="demo-v1",
        )
        controller.set_current_signatures(
            (_signature(), _signature(), _signature()), policy_version=0
        )
        first = controller.decide(
            base_is_rl=False,
            rl_available=True,
            region=0,
            step=0,
            remaining_region_steps=12,
        )
        controller.reserve_probe(first, start_step=0)
        first_chunk = controller.record_autonomous_execution(
            region=0,
            signature=controller.current_signature(0),
            start_step=0,
            duration=5,
            task_return=0.0,
            termination_reason="horizon_reached",
            success=False,
            episode_terminated=False,
            episode_truncated=False,
            probe_decision=first,
            final_region=0,
            path_deviation_m=0.01,
        )

        self.assertTrue(controller.probe_session_active)
        self.assertTrue(first_chunk.was_probe)
        continuation = controller.decide(
            base_is_rl=False,
            rl_available=True,
            region=0,
            step=5,
            remaining_region_steps=7,
        )
        self.assertTrue(continuation.override)
        self.assertTrue(continuation.session_continuation)
        self.assertEqual(continuation.reason, "session_continue")
        controller.reserve_probe(continuation, start_step=5)
        completed = controller.record_autonomous_execution(
            region=0,
            signature=controller.current_signature(0),
            start_step=5,
            duration=5,
            task_return=0.0,
            termination_reason="horizon_reached",
            success=False,
            episode_terminated=False,
            episode_truncated=False,
            probe_decision=continuation,
            final_region=0,
            path_deviation_m=0.01,
        )

        self.assertFalse(controller.probe_session_active)
        self.assertEqual(completed.duration, 10)
        self.assertTrue(completed.success)
        self.assertEqual(controller.current_horizon_steps(0), 15)

    def test_budget_truncation_does_not_grow_horizon(self):
        controller = PolicyChangeProbeController(
            _config(budget_steps=5, initial_horizon_steps=10, required_passes=1),
            anchor_fingerprint="demo-v1",
        )
        controller.set_current_signatures(
            (_signature(), _signature(), _signature()), policy_version=0
        )
        decision = controller.decide(
            base_is_rl=False, rl_available=True, region=0, step=0
        )
        controller.reserve_probe(decision, start_step=0)
        outcome = controller.record_autonomous_execution(
            region=0,
            signature=controller.current_signature(0),
            start_step=0,
            duration=5,
            task_return=0.0,
            termination_reason="horizon_reached",
            success=False,
            episode_terminated=False,
            episode_truncated=False,
            probe_decision=decision,
            final_region=0,
            path_deviation_m=0.01,
        )

        self.assertFalse(controller.probe_session_active)
        self.assertEqual(outcome.termination_reason, "budget_exhausted")
        self.assertEqual(controller.current_horizon_steps(0), 10)

    def test_intermediate_path_deviation_fails_session(self):
        controller = PolicyChangeProbeController(
            _config(initial_horizon_steps=5), anchor_fingerprint="demo-v1"
        )
        controller.set_current_signatures(
            (_signature(), _signature(), _signature()), policy_version=0
        )
        decision = controller.decide(
            base_is_rl=False, rl_available=True, region=0, step=0
        )
        controller.reserve_probe(decision, start_step=0)
        controller.observe_probe_path(0.09)
        outcome = controller.record_autonomous_execution(
            region=0,
            signature=controller.current_signature(0),
            start_step=0,
            duration=5,
            task_return=0.0,
            termination_reason="horizon_reached",
            success=False,
            episode_terminated=False,
            episode_truncated=False,
            probe_decision=decision,
            final_region=0,
            path_deviation_m=0.01,
        )

        self.assertEqual(outcome.termination_reason, "off_path")
        self.assertFalse(outcome.success)


if __name__ == "__main__":
    unittest.main()
