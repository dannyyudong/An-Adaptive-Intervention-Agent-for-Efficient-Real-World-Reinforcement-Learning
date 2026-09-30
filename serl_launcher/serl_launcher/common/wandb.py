import datetime
import hashlib
import tempfile
from copy import copy
from socket import gethostname

import absl.flags as flags
import ml_collections
import wandb


MAX_WANDB_NAME_LENGTH = 128


def _recursive_flatten_dict(d: dict):
    keys, values = [], []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            keys.append(key)
            values.append(value)
    return keys, values


def _make_wandb_experiment_id(exp_descriptor: str, unique_identifier: str) -> str:
    experiment_id = f"{exp_descriptor}_{unique_identifier}"
    if len(experiment_id) <= MAX_WANDB_NAME_LENGTH:
        return experiment_id

    suffix = f"_{unique_identifier}"
    digest = hashlib.sha1(experiment_id.encode("utf-8")).hexdigest()[:8]
    hash_suffix = f"_{digest}{suffix}"
    descriptor_length = MAX_WANDB_NAME_LENGTH - len(hash_suffix)
    if descriptor_length <= 0:
        return experiment_id[: MAX_WANDB_NAME_LENGTH - 9] + f"_{digest}"
    return f"{exp_descriptor[:descriptor_length]}{hash_suffix}"


class WandBLogger(object):
    @staticmethod
    def get_default_config():
        config = ml_collections.ConfigDict()
        config.project = "serl_launcher"  # WandB Project Name
        config.entity = ml_collections.config_dict.FieldReference(None, field_type=str)
        # Which entity to log as (default: your own user)
        config.exp_descriptor = ""  # Run name (doesn't have to be unique)
        # Unique identifier for run (will be automatically generated unless
        # provided)
        config.unique_identifier = ""
        config.group = None
        return config

    def __init__(
        self,
        wandb_config,
        variant,
        wandb_output_dir=None,
        debug=False,
    ):
        self.config = wandb_config
        if self.config.unique_identifier == "":
            self.config.unique_identifier = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

        self.config.experiment_id = self.experiment_id = _make_wandb_experiment_id(
            self.config.exp_descriptor,
            self.config.unique_identifier,
        )

        print(self.config)

        if wandb_output_dir is None:
            wandb_output_dir = tempfile.mkdtemp()

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if debug:
            mode = "disabled"
        else:
            mode = "online"

        self.run = wandb.init(
            config=self._variant,
            project=self.config.project,
            entity=self.config.entity,
            group=self.config.group,
            tags=self.config.tag,
            dir=wandb_output_dir,
            id=self.config.experiment_id,
            save_code=True,
            mode=mode,
        )

        if flags.FLAGS.is_parsed():
            flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
        else:
            flag_dict = {}
        for k in flag_dict:
            if isinstance(flag_dict[k], ml_collections.ConfigDict):
                flag_dict[k] = flag_dict[k].to_dict()
        wandb.config.update(flag_dict)

    def log(self, data: dict, step: int = None):
        data_flat = _recursive_flatten_dict(data)
        data = {k: v for k, v in zip(*data_flat)}
        wandb.log(data, step=step)
