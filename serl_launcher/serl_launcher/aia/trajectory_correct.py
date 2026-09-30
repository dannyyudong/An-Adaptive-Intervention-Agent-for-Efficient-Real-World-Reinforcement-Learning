"""AutoSERL-style expert-trajectory target selection.

The original AutoSERL implementation performs automatic action replacement in
an environment wrapper. Here the geometric part is isolated so a high-level
scheduler can explicitly choose when trajectory correction owns control.
"""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R


TCPPoseExtractor = Callable[[Any], np.ndarray]


@dataclass(frozen=True)
class CorrectionTarget:
    """One expert-trajectory pose selected as a correction target."""

    index: int
    pose: np.ndarray
    translation_distance: float
    rotation_distance: float
    confidence: float


@dataclass(frozen=True)
class TrajectoryProgress:
    """A monotonic projection onto one expert TCP trajectory."""

    index: int
    progress: float
    translation_distance: float
    rotation_distance: float
    arc_length_m: float = 0.0


@dataclass(frozen=True)
class TrajectoryProgressMilestone:
    """Expert-path progress expected after a fixed number of demo steps."""

    index: int
    progress: float
    arc_length_m: float = 0.0


def trajectory_correction_gripper_action(policy_action: np.ndarray) -> float:
    """Return the learned gripper command, or neutral for a fixed-gripper policy."""
    action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
    if action.size not in (6, 7):
        raise RuntimeError(
            "Trajectory correction requires a 6-D fixed-gripper or 7-D "
            f"learned-gripper policy action, got {action.shape}"
        )
    return 0.0 if action.size == 6 else float(action[6])


def resolve_leading_motion_start_index(
    expert_tcp_poses: np.ndarray,
    step_offset: int,
    motion_floor_m: float,
) -> int:
    """Return the last idle pose before the first meaningful local motion.

    The returned index is the earliest expert pose whose forward
    ``step_offset``-step window contains at least ``motion_floor_m`` of path
    arc.  If the complete trajectory is stationary at that scale, index zero
    is retained so callers keep the original conservative behavior.
    """

    poses = np.asarray(expert_tcp_poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[0] == 0 or poses.shape[1] != 7:
        raise ValueError(
            "expert_tcp_poses must have shape (T, 7), " f"got {poses.shape}"
        )
    poses = np.asarray(
        [_validate_tcp_pose(pose, name="expert TCP pose") for pose in poses],
        dtype=np.float32,
    )
    step_offset = int(step_offset)
    motion_floor_m = float(motion_floor_m)
    if step_offset <= 0:
        raise ValueError("step_offset must be positive")
    if not np.isfinite(motion_floor_m) or motion_floor_m < 0.0:
        raise ValueError("motion_floor_m must be finite and non-negative")
    if motion_floor_m == 0.0:
        return 0

    segment_lengths = np.asarray(
        np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1),
        dtype=np.float64,
    )
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(segment_lengths))
    )
    last_index = len(poses) - 1
    for start_index in range(last_index):
        target_index = min(start_index + step_offset, last_index)
        local_motion_m = float(cumulative[target_index] - cumulative[start_index])
        if local_motion_m + 1e-12 >= motion_floor_m:
            return start_index
    return 0


class ExpertTrajectoryProgressEstimator:
    """Project TCP poses continuously onto expert-path arc length.

    This estimator advances only when the measured TCP pose projects onto a
    later point or segment of the expert path. The correction connector and RL
    probe can therefore share the same state-based progress semantics without
    coupling progress to the policy source or elapsed control steps. The
    monotonic local search preserves sequence at self-intersections, while
    segment projection avoids coupling progress to demonstration sampling
    density.
    """

    def __init__(
        self,
        expert_tcp_poses: np.ndarray,
        *,
        lookahead: Optional[int] = None,
        initial_search_steps: Optional[int] = None,
        max_index_advance: Optional[int] = None,
        rotation_weight: float = 0.0,
        initial_index: int = 0,
        max_arc_advance_ratio: Optional[float] = None,
        arc_advance_slack_m: float = 0.0,
        motion_epsilon_m: float = 0.0,
    ) -> None:
        poses = np.asarray(expert_tcp_poses, dtype=np.float32)
        if poses.ndim != 2 or poses.shape[0] == 0 or poses.shape[1] != 7:
            raise ValueError(
                "expert_tcp_poses must have shape (T, 7), " f"got {poses.shape}"
            )
        self.expert_tcp_poses = np.asarray(
            [_validate_tcp_pose(pose, name="expert TCP pose") for pose in poses],
            dtype=np.float32,
        )
        if lookahead is not None and int(lookahead) <= 0:
            raise ValueError("lookahead must be positive when provided")
        if initial_search_steps is not None and int(initial_search_steps) <= 0:
            raise ValueError("initial_search_steps must be positive when provided")
        if max_index_advance is not None and int(max_index_advance) <= 0:
            raise ValueError("max_index_advance must be positive when provided")
        if not np.isfinite(rotation_weight) or float(rotation_weight) < 0.0:
            raise ValueError("rotation_weight must be finite and non-negative")
        initial_index = int(initial_index)
        if not 0 <= initial_index < len(self.expert_tcp_poses):
            raise ValueError(
                f"initial_index must be in [0, {len(self.expert_tcp_poses)}), "
                f"got {initial_index}"
            )
        if max_arc_advance_ratio is not None and (
            not np.isfinite(max_arc_advance_ratio)
            or float(max_arc_advance_ratio) <= 0.0
        ):
            raise ValueError(
                "max_arc_advance_ratio must be finite and positive when provided"
            )
        if not np.isfinite(arc_advance_slack_m) or float(arc_advance_slack_m) < 0.0:
            raise ValueError("arc_advance_slack_m must be finite and non-negative")
        if not np.isfinite(motion_epsilon_m) or float(motion_epsilon_m) < 0.0:
            raise ValueError("motion_epsilon_m must be finite and non-negative")
        self.lookahead = None if lookahead is None else int(lookahead)
        self.initial_search_steps = (
            None if initial_search_steps is None else int(initial_search_steps)
        )
        self.max_index_advance = (
            None if max_index_advance is None else int(max_index_advance)
        )
        self.rotation_weight = float(rotation_weight)
        self.initial_index = initial_index
        self.max_arc_advance_ratio = (
            None if max_arc_advance_ratio is None else float(max_arc_advance_ratio)
        )
        self.arc_advance_slack_m = float(arc_advance_slack_m)
        self.motion_epsilon_m = float(motion_epsilon_m)

        segment_lengths = np.asarray(
            np.linalg.norm(np.diff(self.expert_tcp_poses[:, :3], axis=0), axis=1),
            dtype=np.float64,
        )
        cumulative = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(segment_lengths))
        )
        total = float(cumulative[-1])
        self._segment_lengths_m = segment_lengths
        self._cumulative_arc_length_m = cumulative
        self.total_arc_length_m = total
        self._normalized_arc_length = (
            np.zeros_like(cumulative) if total <= 1e-8 else cumulative / total
        )
        self._last_index: Optional[int] = None
        self._last_arc_length_m: Optional[float] = None
        self._last_tcp_pose: Optional[np.ndarray] = None

    @property
    def last_index(self) -> Optional[int]:
        return self._last_index

    @property
    def last_arc_length_m(self) -> Optional[float]:
        return self._last_arc_length_m

    def reset(self) -> None:
        self._last_index = None
        self._last_arc_length_m = None
        self._last_tcp_pose = None

    def milestone_after_steps(
        self,
        start_index: int,
        step_offset: int,
    ) -> TrajectoryProgressMilestone:
        """Return the expert waypoint reached after ``step_offset`` steps.

        Demonstrations and actor rollouts both contain one sample per
        environment step.  Using the demonstration index keeps the curriculum
        sensitive to locally slow and fast parts of the expert trajectory,
        while the returned arc-length progress remains normalized to [0, 1].
        """

        start_index = int(start_index)
        step_offset = int(step_offset)
        if not 0 <= start_index < len(self.expert_tcp_poses):
            raise ValueError(
                f"start_index must be in [0, {len(self.expert_tcp_poses)}), "
                f"got {start_index}"
            )
        if step_offset < 0:
            raise ValueError("step_offset must be non-negative")
        target_index = min(
            start_index + step_offset,
            len(self.expert_tcp_poses) - 1,
        )
        return TrajectoryProgressMilestone(
            index=target_index,
            progress=float(self._normalized_arc_length[target_index]),
            arc_length_m=float(self._cumulative_arc_length_m[target_index]),
        )

    @staticmethod
    def _interpolate_quaternion(
        start_quaternion: np.ndarray,
        end_quaternion: np.ndarray,
        fraction: float,
    ) -> np.ndarray:
        """Return a stable shortest-path quaternion interpolation."""

        start_quaternion = np.asarray(start_quaternion, dtype=np.float64)
        end_quaternion = np.asarray(end_quaternion, dtype=np.float64)
        if float(np.dot(start_quaternion, end_quaternion)) < 0.0:
            end_quaternion = -end_quaternion
        quaternion = (1.0 - float(fraction)) * start_quaternion + float(
            fraction
        ) * end_quaternion
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12:
            return start_quaternion
        return quaternion / norm

    def project(self, current_tcp_pose: np.ndarray) -> TrajectoryProgress:
        current_pose = _validate_tcp_pose(current_tcp_pose, name="current TCP pose")
        if self._last_index is None:
            start = self.initial_index
            stop = len(self.expert_tcp_poses)
            if self.initial_search_steps is not None:
                stop = min(stop, start + self.initial_search_steps)
        else:
            start = int(self._last_index)
            stop = len(self.expert_tcp_poses)
            forward_limits = [
                limit
                for limit in (self.lookahead, self.max_index_advance)
                if limit is not None
            ]
            if forward_limits:
                stop = min(stop, start + min(forward_limits) + 1)

        minimum_arc = (
            float(self._cumulative_arc_length_m[self.initial_index])
            if self._last_arc_length_m is None
            else float(self._last_arc_length_m)
        )
        maximum_arc = float("inf")
        if self._last_tcp_pose is not None and self.max_arc_advance_ratio is not None:
            tcp_step_distance_m = float(
                np.linalg.norm(current_pose[:3] - self._last_tcp_pose[:3])
            )
            if tcp_step_distance_m <= self.motion_epsilon_m:
                maximum_arc = minimum_arc
            else:
                maximum_arc = min(
                    self.total_arc_length_m,
                    minimum_arc
                    + self.max_arc_advance_ratio * tcp_step_distance_m
                    + self.arc_advance_slack_m,
                )
        projection_candidates = []

        def append_candidate(index, arc_length_m, projected_pose):
            translation_distance = float(
                np.linalg.norm(projected_pose[:3] - current_pose[:3])
            )
            rotation_distance = float(
                (
                    R.from_quat(projected_pose[3:7]).inv()
                    * R.from_quat(current_pose[3:7])
                ).magnitude()
            )
            matching_cost = (
                translation_distance + self.rotation_weight * rotation_distance
            )
            projection_candidates.append(
                (
                    matching_cost,
                    float(arc_length_m),
                    int(index),
                    translation_distance,
                    rotation_distance,
                )
            )

        for index in range(start, stop):
            arc_length_m = float(self._cumulative_arc_length_m[index])
            if (
                arc_length_m + 1e-12 >= minimum_arc
                and arc_length_m <= maximum_arc + 1e-12
            ):
                append_candidate(
                    index,
                    arc_length_m,
                    self.expert_tcp_poses[index],
                )

        for segment_index in range(start, max(start, stop - 1)):
            segment_length_m = float(self._segment_lengths_m[segment_index])
            if segment_length_m <= 1e-12:
                continue
            segment_start_arc = float(self._cumulative_arc_length_m[segment_index])
            segment_end_arc = float(self._cumulative_arc_length_m[segment_index + 1])
            if segment_end_arc + 1e-12 < minimum_arc:
                continue
            if segment_start_arc > maximum_arc + 1e-12:
                continue

            start_xyz = self.expert_tcp_poses[segment_index, :3]
            segment_xyz = self.expert_tcp_poses[segment_index + 1, :3] - start_xyz
            fraction = float(
                np.dot(current_pose[:3] - start_xyz, segment_xyz)
                / (segment_length_m * segment_length_m)
            )
            minimum_fraction = max(
                0.0,
                (minimum_arc - segment_start_arc) / segment_length_m,
            )
            maximum_fraction = min(
                1.0,
                (maximum_arc - segment_start_arc) / segment_length_m,
            )
            if minimum_fraction > maximum_fraction + 1e-12:
                continue
            fraction = float(np.clip(fraction, minimum_fraction, maximum_fraction))
            projected_pose = np.empty((7,), dtype=np.float64)
            projected_pose[:3] = start_xyz + fraction * segment_xyz
            projected_pose[3:7] = self._interpolate_quaternion(
                self.expert_tcp_poses[segment_index, 3:7],
                self.expert_tcp_poses[segment_index + 1, 3:7],
                fraction,
            )
            arc_length_m = segment_start_arc + fraction * segment_length_m
            index = segment_index + 1 if fraction >= 1.0 - 1e-9 else segment_index
            append_candidate(index, arc_length_m, projected_pose)

        if not projection_candidates:
            raise RuntimeError("No monotonic expert trajectory projection candidate")

        minimum_cost = min(candidate[0] for candidate in projection_candidates)
        tied_candidates = [
            candidate
            for candidate in projection_candidates
            if np.isclose(candidate[0], minimum_cost, rtol=0.0, atol=1e-12)
        ]
        _, arc_length_m, index, translation_distance, rotation_distance = max(
            tied_candidates,
            key=lambda candidate: (candidate[1], candidate[2]),
        )
        self._last_index = max(
            int(index),
            self.initial_index if self._last_index is None else int(self._last_index),
        )
        self._last_arc_length_m = max(minimum_arc, float(arc_length_m))
        self._last_tcp_pose = current_pose.copy()
        normalized_progress = (
            0.0
            if self.total_arc_length_m <= 1e-12
            else self._last_arc_length_m / self.total_arc_length_m
        )
        return TrajectoryProgress(
            index=int(self._last_index),
            progress=float(normalized_progress),
            translation_distance=float(translation_distance),
            rotation_distance=float(rotation_distance),
            arc_length_m=float(self._last_arc_length_m),
        )


def _validate_tcp_pose(pose: np.ndarray, *, name: str) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} contains NaN or Inf")
    quaternion_norm = float(np.linalg.norm(pose[3:7]))
    if quaternion_norm <= 1e-8:
        raise ValueError(f"{name} contains a zero quaternion")
    pose = pose.copy()
    pose[3:7] /= quaternion_norm
    return pose


def extract_autoserl_tcp_pose(transition: Any) -> np.ndarray:
    """Extract ``xyz + quaternion`` using the demonstration schema in AutoSERL.

    AutoSERL stores the TCP pose at ``observations.state[0, 4:11]``. Projects
    with a different state layout should pass their own extractor to
    :func:`load_demo_tcp_poses`.
    """

    try:
        state = np.asarray(transition["observations"]["state"])
        pose = state[0, 4:11]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "Transition does not match AutoSERL observations.state[0, 4:11] schema"
        ) from exc
    return _validate_tcp_pose(pose, name="demonstration TCP pose")


def extract_serl_tcp_pose(transition: Any) -> np.ndarray:
    """Extract ``xyz + quaternion`` from the standard SERL robot state.

    After ``SERLObsWrapper`` and ``Quat2MrpWrapper``, fixed-gripper tasks use
    ``tcp_force(3) + tcp_pose(6)``. Learned-gripper tasks prepend
    ``gripper_state(2)`` because Gymnasium's Dict flattening sorts those state
    keys. The TCP pose contains position followed by a modified Rodrigues
    parameter (MRP) orientation, which is converted back to a quaternion here.
    """

    try:
        state = np.asarray(
            transition["observations"]["state"], dtype=np.float32
        ).reshape(-1)
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Transition does not contain observations.state for the SERL schema"
        ) from exc

    if state.shape == (9,):
        tcp_pose = state[3:9]
    elif state.shape == (11,):
        tcp_pose = state[5:11]
    else:
        raise ValueError(
            "SERL observations.state must contain "
            "tcp_force(3) + tcp_pose(6), optionally preceded by "
            f"gripper_state(2); got {state.shape}"
        )

    pose = np.concatenate((tcp_pose[:3], R.from_mrp(tcp_pose[3:6]).as_quat()))
    return _validate_tcp_pose(pose, name="demonstration TCP pose")


def _transform_tcp_poses(
    poses: np.ndarray,
    initial_xyz_euler_pose: Optional[np.ndarray],
) -> np.ndarray:
    """Validate TCP poses and optionally transform them into the world frame."""

    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[0] == 0 or poses.shape[1] != 7:
        raise ValueError(
            f"Demonstration TCP poses must have shape (T, 7), got {poses.shape}"
        )

    poses = np.asarray(
        [_validate_tcp_pose(pose, name="demonstration TCP pose") for pose in poses],
        dtype=np.float32,
    )
    if initial_xyz_euler_pose is None:
        return poses

    initial_pose = np.asarray(initial_xyz_euler_pose, dtype=np.float32).reshape(-1)
    if initial_pose.shape != (6,):
        raise ValueError(
            "initial_xyz_euler_pose must contain [x, y, z, roll, pitch, yaw]"
        )
    initial_rotation = R.from_euler("xyz", initial_pose[3:6])
    poses = poses.copy()
    poses[:, :3] = initial_rotation.apply(poses[:, :3]) + initial_pose[:3]
    poses[:, 3:7] = (initial_rotation * R.from_quat(poses[:, 3:7])).as_quat()
    return poses.astype(np.float32)


def load_demo_tcp_poses(
    demo_path: str | Path,
    *,
    tcp_pose_extractor: TCPPoseExtractor = extract_autoserl_tcp_pose,
    initial_xyz_euler_pose: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Load TCP poses from one pickled demonstration.

    ``initial_xyz_euler_pose`` reproduces AutoSERL's optional conversion from
    demonstration-relative poses into the world frame. It must contain
    ``[x, y, z, roll, pitch, yaw]``. Leave it as ``None`` when poses are already
    expressed in the controller's world frame.
    """

    path = Path(demo_path).expanduser()
    with path.open("rb") as file:
        demonstration = pickle.load(file)

    if isinstance(demonstration, dict) and "transitions" in demonstration:
        demonstration = demonstration["transitions"]
    if not isinstance(demonstration, Iterable):
        raise ValueError(f"Demonstration at {path} is not an iterable of transitions")

    poses = np.asarray(
        [tcp_pose_extractor(transition) for transition in demonstration],
        dtype=np.float32,
    )
    return _transform_tcp_poses(poses, initial_xyz_euler_pose)


def load_demo_tcp_pose_episodes(
    demo_path: str | Path,
    *,
    tcp_pose_extractor: TCPPoseExtractor = extract_serl_tcp_pose,
    initial_xyz_euler_pose: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """Load a pickled SERL demonstration and split it at ``dones``.

    Each returned array represents one episode and has shape ``(T, 7)``. A
    non-terminal tail is retained as the final episode, which also makes the
    loader useful for partially recorded demonstrations.
    """

    path = Path(demo_path).expanduser()
    with path.open("rb") as file:
        demonstration = pickle.load(file)

    if isinstance(demonstration, dict) and "transitions" in demonstration:
        demonstration = demonstration["transitions"]
    if not isinstance(demonstration, Iterable):
        raise ValueError(f"Demonstration at {path} is not an iterable of transitions")

    episodes: list[np.ndarray] = []
    current_poses: list[np.ndarray] = []
    for transition_index, transition in enumerate(demonstration):
        current_poses.append(tcp_pose_extractor(transition))
        try:
            done_values = np.asarray(transition["dones"]).reshape(-1)
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Transition {transition_index} does not contain a dones value"
            ) from exc
        if done_values.size != 1:
            raise ValueError(
                f"Transition {transition_index} dones must be scalar, got shape "
                f"{np.asarray(transition['dones']).shape}"
            )
        if bool(done_values[0]):
            episodes.append(_transform_tcp_poses(current_poses, initial_xyz_euler_pose))
            current_poses = []

    if current_poses:
        episodes.append(_transform_tcp_poses(current_poses, initial_xyz_euler_pose))
    if not episodes:
        raise ValueError(f"Demonstration at {path} contains no transitions")
    return episodes


class AutoSERLTrajectoryConnector:
    """Select correction targets from an actual-state-aligned trajectory window.

    This class has no environment dependency. The actor supplies the current TCP
    pose after every executed environment step and uses the returned target with
    its robot-specific controller. Progress therefore follows measured robot
    motion regardless of whether RL, trajectory correction, CodePolicy, or a
    human intervention produced that motion.
    """

    def __init__(
        self,
        demo_tcp_poses: np.ndarray,
        *,
        window_length: int,
        trigger_threshold: float = 0.02,
        target_threshold: float = 0.005,
        rotation_trigger_threshold: Optional[float] = None,
        rotation_target_threshold: Optional[float] = None,
        max_connection_distance: Optional[float] = None,
        require_forward_direction: bool = True,
        progress_lookahead: Optional[int] = None,
        progress_rotation_weight: float = 0.01,
        progress_max_arc_advance_ratio: Optional[float] = 1.5,
        progress_arc_advance_slack_m: float = 0.002,
        progress_motion_epsilon_m: float = 0.0001,
    ) -> None:
        poses = np.asarray(demo_tcp_poses, dtype=np.float32)
        if poses.ndim != 2 or poses.shape[0] == 0 or poses.shape[1] != 7:
            raise ValueError(
                f"demo_tcp_poses must have shape (T, 7), got {poses.shape}"
            )
        self.demo_tcp_poses = np.asarray(
            [_validate_tcp_pose(pose, name="demonstration TCP pose") for pose in poses],
            dtype=np.float32,
        )
        if window_length <= 0:
            raise ValueError(f"window_length must be positive, got {window_length}")
        if target_threshold < 0.0:
            raise ValueError("target_threshold must be non-negative")
        if trigger_threshold <= target_threshold:
            raise ValueError("trigger_threshold must be greater than target_threshold")
        if (rotation_trigger_threshold is None) != (rotation_target_threshold is None):
            raise ValueError(
                "rotation trigger and target thresholds must be provided together"
            )
        if rotation_trigger_threshold is not None:
            if not np.isfinite(rotation_trigger_threshold) or not np.isfinite(
                rotation_target_threshold
            ):
                raise ValueError("rotation thresholds must be finite")
            if rotation_target_threshold < 0.0:
                raise ValueError("rotation_target_threshold must be non-negative")
            if rotation_trigger_threshold <= rotation_target_threshold:
                raise ValueError(
                    "rotation_trigger_threshold must be greater than "
                    "rotation_target_threshold"
                )
        if (
            max_connection_distance is not None
            and max_connection_distance <= trigger_threshold
        ):
            raise ValueError(
                "max_connection_distance must be greater than trigger_threshold"
            )

        self.window_length = min(int(window_length), len(self.demo_tcp_poses))
        self.trigger_threshold = float(trigger_threshold)
        self.target_threshold = float(target_threshold)
        self.rotation_trigger_threshold = (
            None
            if rotation_trigger_threshold is None
            else float(rotation_trigger_threshold)
        )
        self.rotation_target_threshold = (
            None
            if rotation_target_threshold is None
            else float(rotation_target_threshold)
        )
        self.max_connection_distance = (
            None if max_connection_distance is None else float(max_connection_distance)
        )
        self.require_forward_direction = bool(require_forward_direction)
        self.progress_estimator = ExpertTrajectoryProgressEstimator(
            self.demo_tcp_poses,
            lookahead=(
                self.window_length
                if progress_lookahead is None
                else int(progress_lookahead)
            ),
            initial_search_steps=1,
            rotation_weight=float(progress_rotation_weight),
            max_arc_advance_ratio=progress_max_arc_advance_ratio,
            arc_advance_slack_m=float(progress_arc_advance_slack_m),
            motion_epsilon_m=float(progress_motion_epsilon_m),
        )
        self.latest_progress: Optional[TrajectoryProgress] = None
        self.window_start = 0

    @property
    def window_stop(self) -> int:
        return min(self.window_start + self.window_length, len(self.demo_tcp_poses))

    def reset(self) -> None:
        self.progress_estimator.reset()
        self.latest_progress = None
        self.window_start = 0

    def observe(self, current_tcp_pose: np.ndarray) -> TrajectoryProgress:
        """Align correction progress to the robot's measured TCP pose.

        Call this exactly once after each real environment step. The estimator
        is monotonic within an episode and limits arc advancement using measured
        TCP displacement, so repeated stationary states cannot advance progress.
        """

        progress = self.progress_estimator.project(current_tcp_pose)
        self.latest_progress = progress
        self.window_start = min(
            max(self.window_start, int(progress.index)),
            len(self.demo_tcp_poses) - 1,
        )
        return progress

    def advance_window(self, steps: int = 1) -> None:
        """Explicitly advance the window for legacy callers.

        Runtime actor progress should use :meth:`observe` so it follows measured
        robot state rather than elapsed control steps.
        """

        if steps < 0:
            raise ValueError("steps must be non-negative")
        max_start = max(0, len(self.demo_tcp_poses) - self.window_length)
        self.window_start = min(self.window_start + int(steps), max_start)

    def commit_target(self, target: CorrectionTarget) -> None:
        """Prevent future windows from moving backward past a completed target."""

        self.window_start = min(
            max(self.window_start, int(target.index)),
            len(self.demo_tcp_poses) - 1,
        )

    def nearest_target(self, current_tcp_pose: np.ndarray) -> CorrectionTarget:
        current_pose = _validate_tcp_pose(current_tcp_pose, name="current TCP pose")
        indices = np.arange(self.window_start, self.window_stop)
        candidates = self.demo_tcp_poses[indices]
        translation_distances = np.linalg.norm(
            candidates[:, :3] - current_pose[:3], axis=1
        )
        local_index = int(np.argmin(translation_distances))
        target_index = int(indices[local_index])
        target_pose = candidates[local_index]
        translation_distance = float(translation_distances[local_index])
        rotation_distance = float(
            (
                R.from_quat(target_pose[3:7]).inv() * R.from_quat(current_pose[3:7])
            ).magnitude()
        )

        if self.max_connection_distance is None:
            confidence = 1.0
        else:
            span = self.max_connection_distance - self.trigger_threshold
            confidence = np.clip(
                (self.max_connection_distance - translation_distance) / span,
                0.0,
                1.0,
            )

        return CorrectionTarget(
            index=target_index,
            pose=copy.deepcopy(target_pose),
            translation_distance=translation_distance,
            rotation_distance=rotation_distance,
            confidence=float(confidence),
        )

    def propose(self, current_tcp_pose: np.ndarray) -> Optional[CorrectionTarget]:
        """Return a valid correction target, or ``None`` when correction is unavailable."""

        current_pose = _validate_tcp_pose(current_tcp_pose, name="current TCP pose")
        target = self.nearest_target(current_pose)
        translation_requires_correction = (
            target.translation_distance > self.trigger_threshold
        )
        rotation_requires_correction = bool(
            self.rotation_trigger_threshold is not None
            and target.rotation_distance > self.rotation_trigger_threshold
        )
        if not (translation_requires_correction or rotation_requires_correction):
            return None
        if (
            self.max_connection_distance is not None
            and target.translation_distance > self.max_connection_distance
        ):
            return None
        if self.require_forward_direction and not self._points_forward(
            current_pose, target.index
        ):
            return None
        return target

    def target_reached(
        self, current_tcp_pose: np.ndarray, target: CorrectionTarget
    ) -> bool:
        current_pose = _validate_tcp_pose(current_tcp_pose, name="current TCP pose")
        translation_reached = bool(
            np.linalg.norm(current_pose[:3] - target.pose[:3]) <= self.target_threshold
        )
        if self.rotation_target_threshold is None:
            return translation_reached
        rotation_distance = float(
            (
                R.from_quat(target.pose[3:7]).inv() * R.from_quat(current_pose[3:7])
            ).magnitude()
        )
        return bool(
            translation_reached and rotation_distance <= self.rotation_target_threshold
        )

    def _points_forward(self, current_pose: np.ndarray, target_index: int) -> bool:
        if len(self.demo_tcp_poses) < 2:
            return True
        if target_index >= len(self.demo_tcp_poses) - 1:
            start_index, end_index = target_index - 1, target_index
        else:
            start_index, end_index = target_index, target_index + 1

        demonstration_direction = (
            self.demo_tcp_poses[end_index, :3] - self.demo_tcp_poses[start_index, :3]
        )
        connection_direction = self.demo_tcp_poses[target_index, :3] - current_pose[:3]
        demonstration_norm = float(np.linalg.norm(demonstration_direction))
        connection_norm = float(np.linalg.norm(connection_direction))
        if demonstration_norm <= 1e-8 or connection_norm <= 1e-8:
            return True
        cosine = float(
            np.dot(demonstration_direction, connection_direction)
            / (demonstration_norm * connection_norm)
        )
        return cosine >= 0.0
