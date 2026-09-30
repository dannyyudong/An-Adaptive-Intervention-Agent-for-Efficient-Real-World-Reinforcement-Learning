"""Experiment registry for the public USB-insertion release.

Imports are delayed until a task is selected so listing the registry does not
initialize camera or robot libraries.
"""


def _build_usb_insert_config():
    from experiments.usb_insert.config import TrainConfig

    return TrainConfig()


def _build_aia_usb_insert_config():
    from experiments.aia.usb_insert.config import TrainConfig

    return TrainConfig()


CONFIG_MAPPING = {
    "usb_insert": _build_usb_insert_config,
    "aia_usb_insert": _build_aia_usb_insert_config,
}
