# AIA: Adaptive Intervention Agent

This repository contains the implementation of **AIA**, an adaptive
intervention agent for efficient real-world reinforcement learning.  The
release is built on top of [HIL-SERL](https://github.com/rail-berkeley/hil-serl)
and includes a USB-insertion example that connects the complete path from
demonstrations to learner, actor, high-level option scheduling, evaluation,
and checkpointing.

## What is included

- `serl_launcher/serl_launcher/aia/`: the task-independent AIA scheduler,
  option implementations, replay buffer, probes, state construction, and
  transition logic.
- `examples/train_aia.py`: the distributed AIA learner/actor entry point.
- `examples/experiments/aia/usb_insert/`: the AIA USB-insertion configuration,
  geometry-based CodePolicy, and launch scripts.
- `examples/experiments/usb_insert/`: the underlying USB-insertion environment
  configuration and wrappers.
- `serl_launcher/` and `serl_robot_infra/`: the HIL-SERL learning and robot
  infrastructure required by the example.

The release intentionally does **not** contain datasets, replay buffers,
checkpoints, videos, logs, virtual environments, other paper-task
configurations, or the external AutoSERL baseline experiment tree.

## Installation

The locked environment targets Linux x86-64 and Python 3.10. A CUDA-capable
machine is required for the configured JAX installation.

```bash
git clone git@github.com:dannyyudong/An-Adaptive-Intervention-Agent-for-Efficient-Real-World-Reinforcement-Learning.git
cd An-Adaptive-Intervention-Agent-for-Efficient-Real-World-Reinforcement-Learning
uv sync --frozen
```

Run the offline checks before connecting hardware:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m unittest discover \
  -s serl_launcher/tests -p 'test_*.py' -v
```

## USB-insertion example

The example uses two processes:

1. The learner consumes demonstrations and receives online transitions.
2. The actor owns the robot, cameras, intervention device, and AIA options.

The following values are deliberately not embedded in the repository:

- `DEMO_PATH`: absolute path to a compatible USB-insertion demonstration file.
- `UR_ROBOT_IP`: IP address of the UR controller.
- `HAND_SERIAL` and `EXTERNAL_SERIAL`: RealSense serial numbers.
- `CODE_POLICY_USB_TARGET`: calibrated target position as `x y z` in metres.

The USB CodePolicy included in this release is intentionally a fixed,
geometry-based reference policy. Its current role is to connect the components
and provide a complete, inspectable end-to-end code path. It is not intended to
represent the final task-general agent workflow; the full agent pipeline will
be further updated and refined in subsequent releases. Users must calibrate
`CODE_POLICY_USB_TARGET` for their own setup and validate all motion and safety
parameters before execution.

Before running an actor, review the workspace, reset pose, force limits,
camera crops, action scale, and target coordinates in
`examples/experiments/usb_insert/config.py`. The checked-in values came from
one physical setup and are not safe calibration values for another robot.

Start a learned-scheduler learner:

```bash
DEMO_PATH=/absolute/path/to/usb_insert_demo.pkl \
LEARNED_OPTION_SCHEDULER=1 \
MANUAL_OPTION_SCHEDULER=0 \
uv run bash examples/experiments/aia/usb_insert/run_learner.sh
```

After hardware checks and target calibration, start its actor with the same
run/checkpoint settings:

```bash
UR_ROBOT_IP=192.168.x.x \
HAND_SERIAL=replace_me \
EXTERNAL_SERIAL=replace_me \
CODE_POLICY_USB_TARGET="x y z" \
LEARNED_OPTION_SCHEDULER=1 \
MANUAL_OPTION_SCHEDULER=0 \
uv run bash examples/experiments/aia/usb_insert/run_actor.sh
```

Alternatively, launch both processes in one tmux session:

```bash
NEW_RUN=1 \
DEMO_PATH=/absolute/path/to/usb_insert_demo.pkl \
UR_ROBOT_IP=192.168.x.x \
HAND_SERIAL=replace_me \
EXTERNAL_SERIAL=replace_me \
CODE_POLICY_USB_TARGET="x y z" \
LEARNED_OPTION_SCHEDULER=1 \
MANUAL_OPTION_SCHEDULER=0 \
uv run bash examples/experiments/aia/usb_insert/run_tmux.sh
```

Set `RUN_ID` and `CHECKPOINT_PATH` explicitly when the learner and actor run on
different hosts. Set `RESUME_TRAINING=1` only when intentionally restoring an
existing run. The scripts reject accidental learner reuse of an existing
checkpoint directory.

For the paper ablations, set `AIA_ABLATION` to one of `rl_only`,
`no_recovery`, `no_codepolicy`, `fixed_rule`, or `ours`. Each run records an
`aia_ablation.json` manifest and rejects incompatible resumes.

## Safety and validation boundary

The included tests validate Python contracts, scheduler behavior, USB
CodePolicy geometry, imports, and shell syntax offline. They do not establish
that a new robot setup is calibrated or safe. Robot motion must be preceded by
site-specific workspace, collision, force, camera, gripper, emergency-stop,
and human-intervention checks.

## Acknowledgements and citation

This project is a derivative of HIL-SERL and retains its Apache-2.0 license and
attribution. The trajectory-correction component also documents conceptual and
data-schema compatibility with
[AutoSERL](https://github.com/autoserl/AutoSERL); the AutoSERL baseline
experiment directory is not distributed here. See [NOTICE](NOTICE) for the
provenance boundary.

If you use the underlying HIL-SERL framework, please cite:

```bibtex
@misc{luo2024hilserl,
  title={Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning},
  author={Jianlan Luo and Charles Xu and Jeffrey Wu and Sergey Levine},
  year={2024},
  eprint={2410.21845},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Individual
third-party source files may carry additional compatible notices that remain
in those files.
