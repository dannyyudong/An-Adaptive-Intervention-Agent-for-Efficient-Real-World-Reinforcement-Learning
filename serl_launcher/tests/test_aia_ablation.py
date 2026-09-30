"""Offline contracts for the public AIA ablation interface."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
AIA = ROOT / "examples" / "experiments" / "aia"
SPEC = importlib.util.spec_from_file_location("aia_ablation", AIA / "ablation.py")
AB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AB)


def _config():
    return SimpleNamespace(
        aia_ablation_supported=True,
        scheduler_dqn_exploration_weights=(0.3, 0.5, 0.2),
    )


class AIAAblationTest(unittest.TestCase):
    def test_all_groups_and_weight_override(self):
        for group, allowed in AB.GROUPS.items():
            config = _config()
            manifest = AB.apply_ablation(config, "aia_usb_insert", group)
            self.assertEqual(config.scheduler_allowed_options, allowed)
            np.testing.assert_allclose(manifest["effective_prior"], (0.3, 0.5, 0.2))

        config = _config()
        AB.apply_ablation(config, "aia_usb_insert", "fixed_rule", "6,3,1")
        np.testing.assert_allclose(
            AB.probabilities(config.scheduler_dqn_exploration_weights, (1, 1, 1)),
            (0.6, 0.3, 0.1),
        )

    def test_probability_masks_and_invalid_values(self):
        self.assertEqual(AB.probabilities(None, (1, 0, 1)), (0.5, 0.0, 0.5))
        np.testing.assert_allclose(
            AB.probabilities((0.3, 0.5, 0.2), (1, 0, 1)),
            (0.6, 0.0, 0.4),
        )
        for weights in (
            (0.0, 1.0, 1.0),
            (-1.0, 2.0, 1.0),
            (float("nan"), 1.0, 1.0),
            (float("inf"), 1.0, 1.0),
            (1.0, 2.0),
        ):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                AB.probabilities(weights, (1, 1, 1))
        with self.assertRaises(ValueError):
            AB.probabilities(None, (0, 0, 0))

    def test_scope_and_group_validation(self):
        config = _config()
        self.assertIsNone(AB.apply_ablation(config, "usb_insert", ""))
        for experiment, group in (
            ("usb_insert", "ours"),
            ("aia_usb_insert", "unknown"),
        ):
            with self.subTest(experiment=experiment, group=group):
                with self.assertRaises(ValueError):
                    AB.apply_ablation(config, experiment, group)

    def test_manifest_resume_contract(self):
        manifest = AB.apply_ablation(_config(), "aia_usb_insert", "ours")
        with tempfile.TemporaryDirectory() as directory:
            AB.check_manifest(directory, manifest)
            AB.check_manifest(directory, manifest, resume=True)
            path = Path(directory) / "aia_ablation.json"
            self.assertEqual(json.loads(path.read_text()), manifest)
            with self.assertRaises(ValueError):
                AB.check_manifest(
                    directory,
                    dict(manifest, group="fixed_rule"),
                    resume=True,
                )

    def test_fresh_resume_is_rejected_without_artifacts(self):
        manifest = AB.apply_ablation(_config(), "aia_usb_insert", "ours")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                AB.check_manifest(directory, manifest, resume=True)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_shell_launchers_and_ablation_routing(self):
        task = AIA / "usb_insert"
        for name in ("run_actor.sh", "run_eval.sh", "run_learner.sh", "run_tmux.sh"):
            subprocess.run(
                ["bash", "-n", str(task / name)],
                check=True,
                capture_output=True,
            )

        for group in AB.GROUPS:
            command = (
                'source "$1"; echo "$LEARNED_OPTION_SCHEDULER,$MANUAL_OPTION_SCHEDULER"'
            )
            result = subprocess.run(
                ["bash", "-c", command, "test", str(AIA / "ablation_env.sh")],
                env={"AIA_ABLATION": group},
                text=True,
                capture_output=True,
                check=True,
            )
            expected = "0,0" if group == "rl_only" else "1,0"
            self.assertEqual(result.stdout.strip(), expected)

    def test_public_mapping_is_lazy_and_only_exposes_usb(self):
        mapping_source = (ROOT / "examples" / "experiments" / "mappings.py").read_text()
        self.assertNotIn(
            "from experiments.usb_insert.config import TrainConfig as",
            mapping_source,
        )
        self.assertIn('"usb_insert": _build_usb_insert_config', mapping_source)
        self.assertIn(
            '"aia_usb_insert": _build_aia_usb_insert_config',
            mapping_source,
        )


if __name__ == "__main__":
    unittest.main()
