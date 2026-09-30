"""Offline geometry tests for the public AIA USB CodePolicy example."""

from types import SimpleNamespace
import unittest

import numpy as np

from examples.experiments.aia.usb_insert.code_policy import (
    USBInsertionPlanProvider,
    usb_stage_reached,
)


def _provider(target=(0.1, -0.2, 0.15)):
    return USBInsertionPlanProvider(
        target_xyz=np.asarray(target, dtype=np.float32),
        current_pose_fn=lambda: np.asarray(
            (0.1, -0.2, 0.2, 0.0, 0.0, 0.0, 1.0), dtype=np.float32
        ),
        approach_dz=0.02,
        position_tolerance=0.002,
        rotation_tolerance=0.03,
        move_max_steps=50,
        insert_max_steps=40,
        workspace_low=np.asarray((0.0, -0.3, 0.1), dtype=np.float32),
        workspace_high=np.asarray((0.2, -0.1, 0.3), dtype=np.float32),
        contact_stop_force_z=4.5,
    )


class USBInsertionPlanProviderTest(unittest.TestCase):
    def test_plan_moves_above_target_then_inserts(self):
        provider = _provider()
        plan = provider.build_plan({})

        self.assertIsNotNone(plan)
        self.assertEqual(
            [stage.name for stage in plan.stages],
            ["move_above_usb_target", "insert_usb_downward"],
        )
        np.testing.assert_allclose(
            plan.stages[0].target_xyz - plan.stages[1].target_xyz,
            (0.0, 0.0, 0.02),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            plan.stages[0].target_quat,
            (0.0, 0.0, 0.0, 1.0),
        )
        self.assertTrue(plan.stages[1].metadata["contact_stop"])

    def test_missing_or_out_of_workspace_target_disables_policy(self):
        for target in ((np.nan, np.nan, np.nan), (0.3, -0.2, 0.15)):
            with self.subTest(target=target):
                provider = _provider(target)
                self.assertFalse(provider.available({}))
                self.assertIsNone(provider.build_plan({}))

    def test_invalid_current_pose_is_rejected(self):
        provider = _provider()
        provider.current_pose_fn = lambda: np.zeros(6, dtype=np.float32)
        with self.assertRaises(ValueError):
            provider.build_plan({})

    def test_force_contact_completes_insertion_and_holds(self):
        stage = _provider().build_plan({}).stages[1]
        held = []
        raw_env = SimpleNamespace(
            curr_pos=np.asarray(
                (0.1, -0.2, 0.17, 0.0, 0.0, 0.0, 1.0), dtype=np.float32
            ),
            controller=SimpleNamespace(
                curr_force_lowpass=np.asarray((0.0, 0.0, 5.0), dtype=np.float32)
            ),
            hold_position=lambda: held.append(True),
        )
        env = SimpleNamespace(unwrapped=raw_env)

        self.assertTrue(usb_stage_reached(env, stage))
        self.assertEqual(held, [True])


if __name__ == "__main__":
    unittest.main()
