import ast
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from serl_launcher.aia.probe import (
    AdaptiveRLProbeController,
    RLProbeConfig,
)
from serl_launcher.aia.trajectory_correct import (
    ExpertTrajectoryProgressEstimator,
    TrajectoryProgress,
    resolve_leading_motion_start_index,
)


def _pose(x, y=0.0, z=0.0):
    return np.array([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _progress(value, deviation=0.0, index=0, arc_length_m=None):
    if arc_length_m is None:
        arc_length_m = value
    return TrajectoryProgress(
        index=index,
        progress=float(value),
        translation_distance=float(deviation),
        rotation_distance=0.0,
        arc_length_m=float(arc_length_m),
    )


class ExpertTrajectoryProgressEstimatorTest(unittest.TestCase):
    def test_leading_stationary_prefix_resolves_last_idle_pose(self):
        poses = np.stack(
            [
                _pose(0.0),
                _pose(0.0),
                _pose(0.0),
                _pose(0.002),
                _pose(0.004),
            ]
        )

        start_index = resolve_leading_motion_start_index(
            poses,
            step_offset=2,
            motion_floor_m=0.001,
        )

        self.assertEqual(start_index, 1)
        self.assertEqual(
            resolve_leading_motion_start_index(
                np.stack([_pose(0.0), _pose(0.0), _pose(0.0)]),
                step_offset=2,
                motion_floor_m=0.001,
            ),
            0,
        )

    def test_initial_index_anchors_projection_and_keeps_original_indices(self):
        poses = np.stack([_pose(float(index)) for index in range(5)])
        estimator = ExpertTrajectoryProgressEstimator(
            poses,
            initial_index=2,
            initial_search_steps=1,
        )

        projection = estimator.project(_pose(0.0))
        milestone = estimator.milestone_after_steps(projection.index, 2)

        self.assertEqual(projection.index, 2)
        self.assertAlmostEqual(projection.arc_length_m, 2.0)
        self.assertEqual(milestone.index, 4)
        self.assertAlmostEqual(milestone.arc_length_m, 4.0)

    def test_arc_length_progress_is_monotonic_and_not_time_driven(self):
        poses = np.stack([_pose(0.0), _pose(1.0), _pose(2.0), _pose(3.0)])
        estimator = ExpertTrajectoryProgressEstimator(poses, lookahead=3)

        start = estimator.project(_pose(0.05))
        repeated = estimator.project(_pose(0.05))
        forward = estimator.project(_pose(2.05))
        backward_pose = estimator.project(_pose(1.0))

        self.assertEqual(start.index, 0)
        self.assertEqual(repeated.index, 0)
        self.assertAlmostEqual(repeated.progress, 0.05 / 3.0)
        self.assertAlmostEqual(repeated.arc_length_m, 0.05)
        self.assertEqual(forward.index, 2)
        self.assertAlmostEqual(forward.progress, 2.05 / 3.0)
        self.assertEqual(backward_pose.index, 2)
        self.assertAlmostEqual(backward_pose.progress, forward.progress)
        self.assertAlmostEqual(backward_pose.translation_distance, 1.05)

    def test_reset_allows_a_new_episode_to_match_from_the_start(self):
        poses = np.stack([_pose(0.0), _pose(1.0), _pose(2.0)])
        estimator = ExpertTrajectoryProgressEstimator(poses, lookahead=2)
        self.assertEqual(estimator.project(_pose(2.0)).index, 2)
        estimator.reset()
        self.assertEqual(estimator.project(_pose(0.0)).index, 0)

    def test_milestone_uses_expert_step_offset_and_arc_length_progress(self):
        poses = np.stack([_pose(0.0), _pose(0.1), _pose(0.4), _pose(1.0)])
        estimator = ExpertTrajectoryProgressEstimator(poses)

        milestone = estimator.milestone_after_steps(0, 2)
        capped = estimator.milestone_after_steps(2, 10)

        self.assertEqual(milestone.index, 2)
        self.assertAlmostEqual(milestone.progress, 0.4)
        self.assertAlmostEqual(milestone.arc_length_m, 0.4)
        self.assertEqual(capped.index, 3)
        self.assertAlmostEqual(capped.progress, 1.0)
        self.assertAlmostEqual(capped.arc_length_m, 1.0)

    def test_continuous_progress_is_independent_of_demo_sampling_density(self):
        sparse = ExpertTrajectoryProgressEstimator(np.stack([_pose(0.0), _pose(1.0)]))
        dense = ExpertTrajectoryProgressEstimator(
            np.stack([_pose(value) for value in np.linspace(0.0, 1.0, 21)])
        )

        sparse_projection = sparse.project(_pose(0.437))
        dense_projection = dense.project(_pose(0.437))

        self.assertAlmostEqual(sparse_projection.arc_length_m, 0.437, places=6)
        self.assertAlmostEqual(dense_projection.arc_length_m, 0.437, places=6)
        self.assertAlmostEqual(sparse_projection.progress, 0.437, places=6)
        self.assertAlmostEqual(dense_projection.progress, 0.437, places=6)

    def test_sequence_constraints_track_a_noisy_out_and_back_path(self):
        poses = np.stack(
            [
                _pose(0.0),
                _pose(1.0),
                _pose(2.0),
                _pose(1.001),
                _pose(0.001),
            ]
        )
        estimator = ExpertTrajectoryProgressEstimator(
            poses,
            lookahead=4,
            initial_search_steps=1,
            max_index_advance=1,
            rotation_weight=0.01,
        )

        projections = [
            estimator.project(current_pose)
            for current_pose in (
                _pose(0.001),
                _pose(1.001),
                _pose(2.0),
                _pose(1.0),
                _pose(0.0),
            )
        ]

        self.assertEqual(
            [projection.index for projection in projections],
            [0, 1, 2, 3, 4],
        )
        self.assertAlmostEqual(projections[0].progress, 0.0)
        self.assertAlmostEqual(projections[-1].progress, 1.0)
        self.assertGreater(
            projections[-1].progress,
            projections[2].progress,
        )

    def test_initial_search_does_not_jump_to_overlapping_return_endpoint(self):
        poses = np.stack([_pose(0.01), _pose(1.0), _pose(0.0)])
        estimator = ExpertTrajectoryProgressEstimator(
            poses,
            lookahead=2,
            initial_search_steps=1,
            max_index_advance=1,
        )

        projection = estimator.project(_pose(0.0))

        self.assertEqual(projection.index, 0)
        self.assertAlmostEqual(projection.progress, 0.0)

    def test_dynamic_arc_limit_allows_fast_motion_across_demo_samples(self):
        poses = np.stack([_pose(index * 0.01) for index in range(5)])
        estimator = ExpertTrajectoryProgressEstimator(
            poses,
            lookahead=4,
            initial_search_steps=1,
            max_arc_advance_ratio=1.5,
            arc_advance_slack_m=0.002,
            motion_epsilon_m=0.0001,
        )
        estimator.project(_pose(0.0))

        projection = estimator.project(_pose(0.025))

        self.assertEqual(projection.index, 2)
        self.assertAlmostEqual(projection.arc_length_m, 0.025, places=6)

    def test_dynamic_arc_limit_preserves_out_and_back_sequence(self):
        poses = np.stack(
            [
                _pose(0.0),
                _pose(1.0),
                _pose(2.0),
                _pose(1.001),
                _pose(0.001),
            ]
        )
        estimator = ExpertTrajectoryProgressEstimator(
            poses,
            lookahead=4,
            initial_search_steps=1,
            max_arc_advance_ratio=1.5,
            motion_epsilon_m=0.0001,
        )

        projections = [
            estimator.project(current_pose)
            for current_pose in (
                _pose(0.001),
                _pose(1.001),
                _pose(2.0),
                _pose(1.0),
                _pose(0.0),
            )
        ]

        self.assertEqual(
            [projection.index for projection in projections],
            [0, 1, 2, 3, 4],
        )
        self.assertAlmostEqual(projections[-1].progress, 1.0)

    def test_stationary_tcp_cannot_gain_arc_progress_at_an_overlap(self):
        poses = np.stack([_pose(0.0), _pose(1.0), _pose(0.0)])
        estimator = ExpertTrajectoryProgressEstimator(
            poses,
            lookahead=2,
            initial_search_steps=1,
            max_arc_advance_ratio=1.5,
            arc_advance_slack_m=0.002,
            motion_epsilon_m=0.0001,
        )

        start = estimator.project(_pose(0.0))
        repeated = estimator.project(_pose(0.0))

        self.assertEqual(start.index, 0)
        self.assertEqual(repeated.index, 0)
        self.assertAlmostEqual(repeated.arc_length_m, 0.0)

    def test_reset_clears_dynamic_motion_history_and_reanchors(self):
        poses = np.stack([_pose(0.0), _pose(1.0), _pose(2.0)])
        estimator = ExpertTrajectoryProgressEstimator(
            poses,
            initial_index=1,
            initial_search_steps=1,
            max_arc_advance_ratio=1.5,
        )
        estimator.project(_pose(1.0))
        estimator.project(_pose(2.0))

        estimator.reset()
        projection = estimator.project(_pose(0.0))

        self.assertEqual(projection.index, 1)
        self.assertAlmostEqual(projection.arc_length_m, 1.0)

    def test_rotation_disambiguates_overlapping_forward_candidates(self):
        poses = np.stack([_pose(0.0), _pose(1.0), _pose(1.0)])
        poses[1, 3:7] = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        estimator = ExpertTrajectoryProgressEstimator(
            poses,
            initial_search_steps=1,
            max_index_advance=2,
            rotation_weight=0.01,
        )
        estimator.project(_pose(0.0))

        projection = estimator.project(_pose(1.0))

        self.assertEqual(projection.index, 2)
        self.assertAlmostEqual(projection.rotation_distance, 0.0)


class AdaptiveRLProbeControllerTest(unittest.TestCase):
    def setUp(self):
        self.config = RLProbeConfig(
            initial_steps=5,
            step_increment=5,
            max_steps=15,
            required_passes=2,
            min_progress_delta=0.10,
            expert_progress_fraction=0.8,
            reference_motion_floor_m=0.001,
            stationary_path_tolerance_m=0.005,
            max_path_deviation=0.50,
            stall_steps=100,
            progress_epsilon=1e-4,
            progress_epsilon_m=1e-4,
            step_quantum=5,
            off_path_decrement=5,
            safety_decrement=10,
        )
        self.controller = AdaptiveRLProbeController(self.config)

    def start_probe(
        self,
        progress=None,
        *,
        target_progress=0.20,
        target_index=None,
    ):
        if progress is None:
            progress = _progress(0.0, index=0)
        if target_index is None:
            target_index = progress.index + self.controller.budget_steps
        return self.controller.start_episode(
            progress,
            expert_target_index=target_index,
            expert_target_progress=target_progress,
        )

    def run_qualified_probe(self):
        budget = self.controller.budget_steps
        self.controller.reset_episode()
        self.assertTrue(self.start_probe())
        result = None
        for step in range(1, budget + 1):
            result = self.controller.observe_rl_step(
                _progress(0.20 * step / budget, index=step)
            )
        self.assertIsNotNone(result)
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "budget_reached")
        self.assertTrue(self.controller.mark_episode_completed())
        return result

    def test_two_progress_qualified_probes_grow_budget_by_one_option(self):
        first = self.run_qualified_probe()
        self.assertEqual(first.budget_after, 5)
        self.assertEqual(first.pass_streak, 1)

        second = self.run_qualified_probe()
        self.assertEqual(second.budget_before, 5)
        self.assertEqual(second.budget_after, 10)
        self.assertEqual(second.pass_streak, 0)
        metrics = second.metrics()
        self.assertEqual(metrics["rl_probe/current_budget"], 10)
        self.assertEqual(metrics["rl_probe/budget_delta"], 5)
        self.assertEqual(metrics["rl_probe/budget_grew"], 1)
        self.assertEqual(metrics["rl_probe/budget_shrank"], 0)
        self.assertAlmostEqual(metrics["rl_probe/expert_target_progress"], 0.20)
        self.assertAlmostEqual(metrics["rl_probe/required_progress_delta"], 0.16)
        self.assertAlmostEqual(metrics["rl_probe/progress_ratio"], 1.25)

    def test_budget_growth_is_capped(self):
        for _ in range(6):
            self.run_qualified_probe()
        self.assertEqual(self.controller.budget_steps, 15)

    def test_no_progress_does_not_grow_budget(self):
        self.assertTrue(self.start_probe())
        result = None
        for _ in range(5):
            result = self.controller.observe_rl_step(_progress(0.0))
        self.assertIsNotNone(result)
        self.assertFalse(result.passed)
        self.assertEqual(self.controller.budget_steps, 5)
        self.assertEqual(self.controller.pass_streak, 0)

    def test_expert_milestone_replaces_fixed_minimum_as_pass_threshold(self):
        self.assertTrue(self.start_probe(target_progress=0.50))
        result = None
        for step in range(1, 6):
            result = self.controller.observe_rl_step(
                _progress(0.20 * step / 5, index=step)
            )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.required_progress_delta, 0.40)
        self.assertAlmostEqual(result.progress_delta, 0.20)
        self.assertAlmostEqual(result.progress_ratio, 0.50)
        self.assertFalse(result.passed)

    def test_small_moving_reference_uses_relative_not_global_threshold(self):
        self.assertTrue(self.start_probe(target_progress=0.005))
        result = None
        for step in range(1, 6):
            result = self.controller.observe_rl_step(
                _progress(0.004 * step / 5, index=step)
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.grading_mode, "moving")
        self.assertAlmostEqual(result.required_progress_delta, 0.004)
        self.assertAlmostEqual(result.required_motion_m, 0.004)
        self.assertTrue(result.passed)

    def test_stationary_reference_passes_only_inside_tight_local_corridor(self):
        self.assertTrue(self.start_probe(target_progress=0.0005))
        result = None
        for _ in range(5):
            result = self.controller.observe_rl_step(
                _progress(0.0, deviation=0.004, arc_length_m=0.0)
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.grading_mode, "stationary")
        self.assertAlmostEqual(result.required_motion_m, 0.0)
        self.assertTrue(result.passed)
        self.assertEqual(result.metrics()["rl_probe/grading_mode_stationary"], 1)

    def test_stationary_reference_rejects_drift_inside_broad_safety_corridor(self):
        self.assertTrue(self.start_probe(target_progress=0.0005))
        result = None
        for _ in range(5):
            result = self.controller.observe_rl_step(
                _progress(0.0, deviation=0.006, arc_length_m=0.0)
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.reason, "budget_reached")
        self.assertLess(result.max_path_deviation, self.config.max_path_deviation)
        self.assertFalse(result.passed)

    def test_stationary_reference_is_not_stalled_before_a_long_budget(self):
        controller = AdaptiveRLProbeController(
            replace(self.config, max_steps=15, stall_steps=2)
        )
        state = controller.state_dict()
        state["budget_steps"] = 15
        controller.restore_state(state)
        self.assertTrue(
            controller.start_episode(
                _progress(0.0),
                expert_target_index=15,
                expert_target_progress=0.0005,
            )
        )

        result = None
        for _ in range(15):
            result = controller.observe_rl_step(
                _progress(0.0, deviation=0.001, arc_length_m=0.0)
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.reason, "budget_reached")
        self.assertTrue(result.passed)

    def test_off_path_probe_shrinks_restored_budget(self):
        state = self.controller.state_dict()
        state["budget_steps"] = 15
        self.controller.restore_state(state)
        self.assertTrue(self.start_probe())

        result = self.controller.observe_rl_step(_progress(0.01, deviation=0.60))

        self.assertEqual(result.reason, "off_path")
        self.assertEqual(result.budget_after, 10)
        self.assertFalse(result.passed)
        self.assertEqual(result.metrics()["rl_probe/budget_shrank"], 1)

    def test_intervention_shrinks_faster_but_never_below_initial(self):
        state = self.controller.state_dict()
        state["budget_steps"] = 10
        self.controller.restore_state(state)
        self.assertTrue(self.start_probe())

        result = self.controller.observe_rl_step(_progress(0.01), intervention=True)

        self.assertEqual(result.reason, "intervention")
        self.assertEqual(result.budget_after, 5)

    def test_probe_is_skipped_outside_expert_corridor(self):
        self.assertFalse(self.start_probe(_progress(0.0, deviation=0.75)))
        self.assertFalse(self.controller.active)
        self.assertFalse(self.controller.attempted_this_episode)
        self.assertFalse(self.controller.should_force_rl(rl_available=True))
        self.assertTrue(self.start_probe())

    def test_completed_probe_cannot_restart_in_the_same_episode(self):
        self.assertTrue(self.start_probe())
        result = None
        for step in range(1, 6):
            result = self.controller.observe_rl_step(_progress(0.03 * step))
        self.assertIsNotNone(result)
        self.assertFalse(
            self.start_probe(
                _progress(0.2, index=5),
                target_progress=0.4,
                target_index=10,
            )
        )

    def test_forcing_requires_real_rl_availability(self):
        self.assertTrue(self.start_probe())
        self.assertTrue(self.controller.should_force_rl(rl_available=True))
        self.assertFalse(self.controller.should_force_rl(rl_available=False))

    def test_probe_is_scheduled_on_first_third_and_fifth_episodes(self):
        controller = AdaptiveRLProbeController(replace(self.config, episode_interval=2))
        scheduled = []
        episodes_until_next = []

        for episode in range(1, 6):
            scheduled.append(controller.reset_episode())
            metrics = controller.schedule_metrics()
            self.assertEqual(metrics["rl_probe/episode_counter"], episode)
            episodes_until_next.append(metrics["rl_probe/episodes_until_next"])
            self.assertEqual(
                controller.attempted_this_episode,
                not controller.scheduled_this_episode,
            )
            self.assertTrue(controller.mark_episode_completed())

        self.assertEqual(scheduled, [True, False, True, False, True])
        self.assertEqual(episodes_until_next, [1, 0, 1, 0, 1])

    def test_completed_episode_count_survives_restore(self):
        controller = AdaptiveRLProbeController(replace(self.config, episode_interval=2))
        for _ in range(2):
            controller.reset_episode()
            controller.mark_episode_completed()

        restored = AdaptiveRLProbeController(replace(self.config, episode_interval=2))
        restored.restore_state(controller.state_dict())

        self.assertEqual(restored.completed_episodes, 2)
        self.assertTrue(restored.reset_episode())
        self.assertEqual(restored.episode_counter, 3)


class RLProbeActorIntegrationTest(unittest.TestCase):
    def setUp(self):
        repo_root = Path(__file__).resolve().parents[2]
        self.train_path = repo_root / "examples" / "train_aia.py"
        self.source = self.train_path.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source, filename=str(self.train_path))

    def test_probe_overrides_behavior_without_replacing_physical_mask(self):
        begin_calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "begin"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "scheduler_transition_builder"
        ]
        self.assertEqual(len(begin_calls), 1)
        self.assertGreaterEqual(len(begin_calls[0].args), 3)
        self.assertIsInstance(begin_calls[0].args[2], ast.Name)
        self.assertEqual(begin_calls[0].args[2].id, "available_actions")
        self.assertNotIn("OptionID.RL_PROBE", self.source)

    def test_manual_override_precedes_probe_override(self):
        manual_branch = self.source.index("if manual_option_override_pending:")
        probe_branch = self.source.index("elif rl_probe_forces_rl:", manual_branch)
        learned_branch = self.source.index(
            "FLAGS.learned_option_scheduler", probe_branch
        )
        self.assertLess(manual_branch, probe_branch)
        self.assertLess(probe_branch, learned_branch)

    def test_actor_builds_probe_target_from_expert_step_milestone(self):
        self.assertIn("milestone_after_steps(", self.source)
        self.assertIn("expert_target_progress=milestone.progress", self.source)
        self.assertIn(
            "expert_target_arc_length_m=milestone.arc_length_m",
            self.source,
        )

    def test_default_max_budget_covers_the_complete_expert_trajectory(self):
        resolver_node = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "resolve_rl_probe_max_steps"
        )
        resolver_module = ast.Module(body=[resolver_node], type_ignores=[])
        ast.fix_missing_locations(resolver_module)
        namespace = {}
        exec(compile(resolver_module, str(self.train_path), "exec"), namespace)
        resolve_max_steps = namespace["resolve_rl_probe_max_steps"]

        max_steps = resolve_max_steps(120, 5)
        self.assertEqual(max_steps, 120)
        self.assertEqual(resolve_max_steps(97, 5), 100)
        self.assertEqual(resolve_max_steps(120, 5, 50), 50)

        poses = np.stack([_pose(float(index)) for index in range(120)])
        estimator = ExpertTrajectoryProgressEstimator(poses)
        milestone = estimator.milestone_after_steps(0, max_steps)
        self.assertEqual(milestone.index, 119)
        self.assertAlmostEqual(milestone.progress, 1.0)
        self.assertIn("probe_reference_pose_count", self.source)
        self.assertIn("probe_start_index", self.source)

    def test_actor_disables_pending_probe_on_unscheduled_episodes(self):
        self.assertIn("scheduled = rl_probe_controller.reset_episode()", self.source)
        self.assertIn("if not scheduled:", self.source)
        self.assertIn("rl_probe_controller.mark_episode_completed()", self.source)


if __name__ == "__main__":
    unittest.main()
