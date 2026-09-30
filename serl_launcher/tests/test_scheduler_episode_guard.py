import ast
import unittest
from pathlib import Path

import numpy as np

from serl_launcher.aia.options import OptionID


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = REPO_ROOT / "examples" / "train_aia.py"
TRAIN_TREE = ast.parse(
    TRAIN_PATH.read_text(encoding="utf-8"),
    filename=str(TRAIN_PATH),
)
GUARD_NODE = next(
    node
    for node in TRAIN_TREE.body
    if isinstance(node, ast.ClassDef)
    and node.name == "ConsecutiveCodePolicyEpisodeGuard"
)
GUARD_NAMESPACE = {"np": np, "OptionID": OptionID}
exec(
    compile(
        ast.Module(body=[GUARD_NODE], type_ignores=[]),
        filename=str(TRAIN_PATH),
        mode="exec",
    ),
    GUARD_NAMESPACE,
)
ConsecutiveCodePolicyEpisodeGuard = GUARD_NAMESPACE["ConsecutiveCodePolicyEpisodeGuard"]


class ConsecutiveCodePolicyEpisodeGuardTest(unittest.TestCase):
    def setUp(self):
        self.guard = ConsecutiveCodePolicyEpisodeGuard(
            required_streak=2,
            trajectory_block_steps=20,
        )

    def finish_code_policy_episode(self):
        self.guard.mark_option_started(OptionID.CODE_POLICY)
        return self.guard.finish_episode()

    def test_blocks_first_twenty_steps_of_third_episode(self):
        first = self.finish_code_policy_episode()
        self.assertEqual(first["consecutive_code_policy_episodes"], 1)
        self.assertFalse(self.guard.trajectory_blocked)

        second = self.finish_code_policy_episode()
        self.assertEqual(second["consecutive_code_policy_episodes"], 2)
        self.assertEqual(second["next_episode_trajectory_block_steps"], 20)

        base_mask = np.ones(len(OptionID), dtype=bool)
        for _ in range(20):
            masked = self.guard.apply_action_mask(base_mask)
            self.assertFalse(masked[int(OptionID.TRAJECTORY_CORRECTION)])
            self.assertTrue(masked[int(OptionID.RL)])
            self.assertTrue(masked[int(OptionID.CODE_POLICY)])
            self.guard.record_env_step()

        unmasked = self.guard.apply_action_mask(base_mask)
        self.assertTrue(unmasked[int(OptionID.TRAJECTORY_CORRECTION)])
        self.assertTrue(np.all(base_mask))

    def test_episode_without_code_policy_resets_streak(self):
        self.finish_code_policy_episode()
        summary = self.guard.finish_episode()
        self.assertEqual(summary["consecutive_code_policy_episodes"], 0)
        self.assertEqual(summary["next_episode_trajectory_block_steps"], 0)
        self.assertFalse(self.guard.trajectory_blocked)

    def test_code_policy_in_blocked_episode_extends_guard_to_next_episode(self):
        self.finish_code_policy_episode()
        self.finish_code_policy_episode()
        third = self.finish_code_policy_episode()
        self.assertEqual(third["consecutive_code_policy_episodes"], 3)
        self.assertEqual(third["next_episode_trajectory_block_steps"], 20)
        self.assertTrue(self.guard.trajectory_blocked)


if __name__ == "__main__":
    unittest.main()
