"""Shared training configuration overrides for AIA experiments."""


class AIATrainingConfigMixin:
    """Configuration shared by every AIA task."""

    aia_ablation_supported = True
    aia_ablation = None
    scheduler_allowed_options = (True, True, True)

    scheduler_code_policy_cost: float = 0.05
