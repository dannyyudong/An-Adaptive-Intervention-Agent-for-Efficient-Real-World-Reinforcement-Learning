"""AIA ablation configuration; pure Python, safe to import without hardware."""
import fcntl
import json
import math
import os
from pathlib import Path

GROUPS = {
    "rl_only": (True, False, False),
    "no_recovery": (True, False, True),
    "no_codepolicy": (True, True, False),
    "fixed_rule": (True, True, True),
    "ours": (True, True, True),
}


def probabilities(weights, mask):
    """Same positive exploration prior as SchedulerDQN, masked and normalized."""
    weights = (1.0, 1.0, 1.0) if weights is None else tuple(weights)
    if len(weights) != 3 or len(mask) != 3:
        raise ValueError(
            "Expected three weights/mask entries: RL, Recovery, CodePolicy"
        )
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Exploration weights must be finite and strictly positive")
    values = [w if allowed else 0.0 for w, allowed in zip(weights, mask)]
    total = sum(values)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("No valid weighted Option")
    return tuple(value / total for value in values)


def apply_ablation(config, exp_name, group, weights_override=None):
    """Resolve after task inheritance, including task-specific environment overrides."""
    if not group:
        return None
    if not exp_name.startswith("aia_") or not getattr(
        config, "aia_ablation_supported", False
    ):
        raise ValueError("AIA_ABLATION is only supported by AIA tasks")
    if group not in GROUPS:
        raise ValueError(f"Unknown AIA_ABLATION={group!r}; choose {tuple(GROUPS)}")
    weights = getattr(config, "scheduler_dqn_exploration_weights", None)
    if weights_override is not None:
        weights = tuple(float(value.strip()) for value in weights_override.split(","))
    probabilities(weights, (True, True, True))
    config.scheduler_dqn_exploration_weights = weights
    config.aia_ablation = group
    config.scheduler_allowed_options = GROUPS[group]
    return {
        "schema_version": 1,
        "exp_name": exp_name,
        "group": group,
        "allowed_options": list(GROUPS[group]),
        "exploration_weights": list(weights) if weights is not None else None,
        "effective_prior": list(probabilities(weights, (True, True, True))),
        "rl_probe": bool(getattr(config, "scheduler_rl_probe_enabled", False))
        if group != "rl_only"
        else False,
        "policy_change_probe": bool(
            getattr(config, "scheduler_policy_change_probe_enabled", False)
        )
        if group != "rl_only"
        else False,
    }


def check_manifest(checkpoint_path, manifest, *, resume=False):
    """Lock actor/learner creation and reject cross-group resumes before env creation."""
    if checkpoint_path is None:
        if manifest is not None:
            raise ValueError("Ablations require --checkpoint_path")
        return
    directory = Path(checkpoint_path)
    path = directory / "aia_ablation.json"
    if manifest is None:
        if path.exists():
            raise ValueError("This run requires its saved AIA_ABLATION setting")
        return
    if resume and not path.exists():
        raise ValueError("Cannot resume a legacy run without aia_ablation.json")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".aia_ablation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError(
                    "Ablation task/group/weights/Probe settings differ from saved run"
                )
        else:
            patterns = (
                "checkpoint_*",
                "buffer/*.pkl",
                "demo_buffer/*.pkl",
                "scheduler_buffer/*.pkl",
                "scheduler_checkpoints/*",
            )
            if any(any(directory.glob(pattern)) for pattern in patterns):
                raise ValueError(
                    "Use a fresh directory for an ablation; existing training data found"
                )
            temporary = directory / f".aia_ablation.{os.getpid()}.tmp"
            temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, path)
