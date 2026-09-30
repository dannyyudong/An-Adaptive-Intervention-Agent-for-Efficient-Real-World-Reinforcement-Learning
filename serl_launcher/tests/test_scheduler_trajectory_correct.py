import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from serl_launcher.aia.trajectory_correct import (
    AutoSERLTrajectoryConnector,
    trajectory_correction_gripper_action,
)


def _pose(x=0.0, rotation=None):
    quaternion = (
        R.identity().as_quat()
        if rotation is None
        else R.from_rotvec(np.asarray(rotation, dtype=np.float32)).as_quat()
    )
    return np.asarray([x, 0.0, 0.0, *quaternion], dtype=np.float32)


def _xyz_pose(x=0.0, y=0.0, z=0.0):
    return np.asarray([x, y, z, *R.identity().as_quat()], dtype=np.float32)


class AutoSERLTrajectoryConnectorRotationTest(unittest.TestCase):
    def _connector(self):
        return AutoSERLTrajectoryConnector(
            np.stack([_pose()]),
            window_length=1,
            trigger_threshold=0.02,
            target_threshold=0.005,
            rotation_trigger_threshold=0.20,
            rotation_target_threshold=0.05,
            max_connection_distance=0.08,
            require_forward_direction=False,
        )

    def test_rotation_error_alone_makes_correction_available(self):
        target = self._connector().propose(_pose(rotation=[0.0, 0.0, 0.30]))

        self.assertIsNotNone(target)
        self.assertAlmostEqual(target.translation_distance, 0.0)
        self.assertAlmostEqual(target.rotation_distance, 0.30, places=5)

    def test_target_reached_requires_translation_and_rotation(self):
        connector = self._connector()
        target = connector.nearest_target(_pose(rotation=[0.0, 0.0, 0.30]))

        self.assertFalse(
            connector.target_reached(_pose(rotation=[0.0, 0.0, 0.06]), target)
        )
        self.assertTrue(
            connector.target_reached(_pose(rotation=[0.0, 0.0, 0.04]), target)
        )

    def test_rotation_thresholds_remain_opt_in_for_other_tasks(self):
        connector = AutoSERLTrajectoryConnector(
            np.stack([_pose()]),
            window_length=1,
            trigger_threshold=0.02,
            target_threshold=0.005,
            max_connection_distance=0.08,
            require_forward_direction=False,
        )

        self.assertIsNone(connector.propose(_pose(rotation=[0.0, 0.0, 0.30])))


class TrajectoryCorrectionActionShapeTest(unittest.TestCase):
    def test_fixed_gripper_policy_uses_neutral_command(self):
        self.assertEqual(
            trajectory_correction_gripper_action(np.zeros(6, dtype=np.float32)),
            0.0,
        )

    def test_learned_gripper_policy_preserves_command(self):
        action = np.zeros(7, dtype=np.float32)
        action[6] = 0.75

        self.assertAlmostEqual(trajectory_correction_gripper_action(action), 0.75)

    def test_other_action_shapes_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "6-D fixed-gripper or 7-D"):
            trajectory_correction_gripper_action(np.zeros(3, dtype=np.float32))


class AutoSERLTrajectoryConnectorActualProgressTest(unittest.TestCase):
    def _connector(self, poses, *, window_length=20):
        return AutoSERLTrajectoryConnector(
            np.asarray(poses, dtype=np.float32),
            window_length=window_length,
            trigger_threshold=0.005,
            target_threshold=0.002,
            max_connection_distance=0.08,
            require_forward_direction=False,
        )

    def test_fast_robot_progress_does_not_correct_back_to_stale_high_pose(self):
        demo_z = np.linspace(0.305, 0.263, 96)
        connector = self._connector(
            [_xyz_pose(z=float(z)) for z in demo_z],
        )

        for z in np.linspace(0.304, 0.273, 20):
            connector.observe(_xyz_pose(z=float(z)))

        target = connector.propose(_xyz_pose(x=0.01, z=0.273))

        self.assertIsNotNone(target)
        self.assertGreater(target.index, 70)
        self.assertLess(target.pose[2], 0.280)
        self.assertLess(abs(float(target.pose[2]) - 0.273), 0.003)

    def test_stationary_actual_state_does_not_advance_progress(self):
        connector = self._connector(
            [_pose(float(index) * 0.01) for index in range(10)],
            window_length=3,
        )
        first = connector.observe(_pose(0.0))

        repeated = [connector.observe(_pose(0.0)) for _ in range(5)]

        self.assertEqual(first.index, 0)
        self.assertTrue(all(progress.index == 0 for progress in repeated))
        self.assertEqual(connector.window_start, 0)

    def test_progress_uses_every_observed_robot_state_and_is_monotonic(self):
        connector = self._connector(
            [_pose(float(index) * 0.01) for index in range(10)],
            window_length=3,
        )

        observed = [
            connector.observe(_pose(position))
            for position in (0.0, 0.018, 0.039, 0.061)
        ]
        backward = connector.observe(_pose(0.02))

        self.assertEqual([progress.index for progress in observed], [0, 1, 3, 6])
        self.assertEqual(backward.index, 6)
        self.assertEqual(connector.window_start, 6)

    def test_legitimate_upward_expert_progress_remains_available(self):
        demo_z = np.linspace(0.20, 0.30, 11)
        connector = self._connector(
            [_xyz_pose(z=float(z)) for z in demo_z],
            window_length=3,
        )
        for z in (0.20, 0.22, 0.24, 0.26):
            connector.observe(_xyz_pose(z=z))

        target = connector.propose(_xyz_pose(x=0.01, z=0.26))

        self.assertIsNotNone(target)
        self.assertGreaterEqual(target.index, 6)
        self.assertAlmostEqual(float(target.pose[2]), 0.26, places=5)

    def test_reset_restarts_actual_state_progress(self):
        connector = self._connector(
            [_pose(float(index) * 0.01) for index in range(10)],
            window_length=3,
        )
        for position in (0.0, 0.02, 0.04, 0.06):
            connector.observe(_pose(position))
        self.assertGreaterEqual(connector.window_start, 6)

        connector.reset()
        restarted = connector.observe(_pose(0.0))

        self.assertEqual(restarted.index, 0)
        self.assertEqual(connector.window_start, 0)
        self.assertIs(connector.latest_progress, restarted)


if __name__ == "__main__":
    unittest.main()
