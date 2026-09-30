from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass(frozen=True)
class MarkerDetection:
    marker_id: int
    rvec: np.ndarray
    tvec: np.ndarray
    corners: np.ndarray
    reprojection_error_px: float

    @property
    def T_camera_target(self) -> np.ndarray:
        return transform_from_rvec_tvec(self.rvec, self.tvec)


def aruco_dictionary(name: str):
    if name not in ARUCO_DICTS:
        known = ", ".join(sorted(ARUCO_DICTS))
        raise ValueError(f"Unknown ArUco dictionary {name!r}; expected one of: {known}")
    dict_id = ARUCO_DICTS[name]
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def aruco_parameters():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()


def aruco_detect_markers(gray: np.ndarray, dictionary, params):
    # OpenCV >= 4.7 removed the free function cv2.aruco.detectMarkers in favor
    # of the ArucoDetector class; fall back to the old API on older builds.
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = detector.detectMarkers(gray)
        return corners, ids, rejected
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def estimate_marker_pose(
    corners: np.ndarray,
    marker_length_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # OpenCV >= 4.7 removed estimatePoseSingleMarkers; reproduce it with solvePnP
    # using the same planar marker object points and corner ordering.
    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [np.asarray(corners, dtype=np.float32).reshape(1, 4, 2)],
            float(marker_length_m),
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(dist_coeffs, dtype=np.float64),
        )
        return (
            np.asarray(rvecs[0], dtype=np.float64).reshape(3),
            np.asarray(tvecs[0], dtype=np.float64).reshape(3),
        )
    ok, rvec, tvec = cv2.solvePnP(
        marker_object_points(marker_length_m),
        np.asarray(corners, dtype=np.float64).reshape(4, 2),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed to estimate marker pose")
    return np.asarray(rvec, dtype=np.float64).reshape(3), np.asarray(
        tvec, dtype=np.float64
    ).reshape(3)


def detect_single_marker(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length_m: float,
    marker_id: int,
    dict_name: str = "DICT_4X4_50",
) -> MarkerDetection | None:
    dictionary = aruco_dictionary(dict_name)
    params = aruco_parameters()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detect_markers(gray, dictionary, params)
    if ids is None:
        return None

    ids_flat = ids.reshape(-1)
    matches = np.flatnonzero(ids_flat == int(marker_id))
    if len(matches) == 0:
        return None

    idx = int(matches[0])
    rvec, tvec = estimate_marker_pose(
        corners[idx],
        marker_length_m,
        camera_matrix,
        dist_coeffs,
    )
    err = marker_reprojection_error(
        corners=np.asarray(corners[idx], dtype=np.float64).reshape(4, 2),
        rvec=rvec,
        tvec=tvec,
        marker_length_m=marker_length_m,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    return MarkerDetection(
        marker_id=int(ids_flat[idx]),
        rvec=rvec,
        tvec=tvec,
        corners=np.asarray(corners[idx], dtype=np.float64).reshape(4, 2),
        reprojection_error_px=float(err),
    )


def marker_object_points(marker_length_m: float) -> np.ndarray:
    half = float(marker_length_m) / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def marker_reprojection_error(
    corners: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    marker_length_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        marker_object_points(marker_length_m),
        np.asarray(rvec, dtype=np.float64).reshape(3),
        np.asarray(tvec, dtype=np.float64).reshape(3),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
    )
    projected = projected.reshape(4, 2)
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1))))


def draw_marker_detection(
    image_bgr: np.ndarray,
    detection: MarkerDetection | None,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length_m: float,
) -> np.ndarray:
    out = image_bgr.copy()
    if detection is None:
        return out
    corners = [detection.corners.astype(np.float32).reshape(1, 4, 2)]
    ids = np.array([[detection.marker_id]], dtype=np.int32)
    cv2.aruco.drawDetectedMarkers(out, corners, ids)
    cv2.drawFrameAxes(
        out,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
        detection.rvec.reshape(3, 1),
        detection.tvec.reshape(3, 1),
        float(axis_length_m),
    )
    return out


def transform_from_ur_pose_rotvec(pose: Iterable[float]) -> np.ndarray:
    pose = np.asarray(list(pose), dtype=np.float64).reshape(6)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_rotvec(np.asarray(rvec, dtype=np.float64).reshape(3)).as_matrix()
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def average_transforms(transforms: Iterable[np.ndarray]) -> np.ndarray:
    transforms = [np.asarray(T, dtype=np.float64).reshape(4, 4) for T in transforms]
    if not transforms:
        raise ValueError("Cannot average an empty transform list")
    rotations = R.from_matrix([T[:3, :3] for T in transforms])
    T_mean = np.eye(4, dtype=np.float64)
    T_mean[:3, :3] = rotations.mean().as_matrix()
    T_mean[:3, 3] = np.mean([T[:3, 3] for T in transforms], axis=0)
    return T_mean


def rotation_error_deg(T_delta: np.ndarray) -> float:
    return float(np.degrees(R.from_matrix(T_delta[:3, :3]).magnitude()))


def translation_error_mm(T_delta: np.ndarray) -> float:
    return float(np.linalg.norm(T_delta[:3, 3]) * 1000.0)


def transform_to_jsonable(T: np.ndarray) -> list[list[float]]:
    return np.asarray(T, dtype=np.float64).reshape(4, 4).round(12).tolist()


def transform_from_jsonable(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(4, 4)


def _solve_one_method(
    T_base_tcp: list[np.ndarray],
    T_camera_target: list[np.ndarray],
    method_name: str,
) -> dict[str, Any]:
    method = HAND_EYE_METHODS[method_name]
    T_target_camera = [invert_transform(T) for T in T_camera_target]

    R_tcp2base = [T[:3, :3] for T in T_base_tcp]
    t_tcp2base = [T[:3, 3].reshape(3, 1) for T in T_base_tcp]
    R_target2camera = [T[:3, :3] for T in T_target_camera]
    t_target2camera = [T[:3, 3].reshape(3, 1) for T in T_target_camera]

    R_tcp_target, t_tcp_target = cv2.calibrateHandEye(
        R_tcp2base,
        t_tcp2base,
        R_target2camera,
        t_target2camera,
        method=method,
    )

    T_tcp_target = np.eye(4, dtype=np.float64)
    T_tcp_target[:3, :3] = np.asarray(R_tcp_target, dtype=np.float64).reshape(3, 3)
    T_tcp_target[:3, 3] = np.asarray(t_tcp_target, dtype=np.float64).reshape(3)

    base_camera_samples = [
        T_base_tcp_i @ T_tcp_target @ invert_transform(T_camera_target_i)
        for T_base_tcp_i, T_camera_target_i in zip(T_base_tcp, T_camera_target)
    ]
    T_base_camera = average_transforms(base_camera_samples)

    residuals = []
    for T_base_tcp_i, T_camera_target_i in zip(T_base_tcp, T_camera_target):
        lhs = T_base_tcp_i @ T_tcp_target
        rhs = T_base_camera @ T_camera_target_i
        delta = invert_transform(rhs) @ lhs
        residuals.append(
            {
                "translation_mm": translation_error_mm(delta),
                "rotation_deg": rotation_error_deg(delta),
            }
        )

    trans = np.array([r["translation_mm"] for r in residuals], dtype=np.float64)
    rot = np.array([r["rotation_deg"] for r in residuals], dtype=np.float64)
    return {
        "method": method_name,
        "T_base_camera": T_base_camera,
        "T_tcp_target": T_tcp_target,
        "residuals": residuals,
        "summary": {
            "mean_translation_mm": float(np.mean(trans)),
            "median_translation_mm": float(np.median(trans)),
            "max_translation_mm": float(np.max(trans)),
            "mean_rotation_deg": float(np.mean(rot)),
            "median_rotation_deg": float(np.median(rot)),
            "max_rotation_deg": float(np.max(rot)),
        },
    }


def solve_eye_to_hand(
    T_base_tcp: Iterable[np.ndarray],
    T_camera_target: Iterable[np.ndarray],
    method: str = "PARK",
) -> dict[str, Any]:
    T_base_tcp = [np.asarray(T, dtype=np.float64).reshape(4, 4) for T in T_base_tcp]
    T_camera_target = [
        np.asarray(T, dtype=np.float64).reshape(4, 4) for T in T_camera_target
    ]
    if len(T_base_tcp) != len(T_camera_target):
        raise ValueError("Robot and camera sample counts differ")
    if len(T_base_tcp) < 3:
        raise ValueError(
            "At least 3 samples are required; 15-25 diverse poses are recommended"
        )

    method = method.upper()
    methods = list(HAND_EYE_METHODS) if method == "ALL" else [method]
    if any(name not in HAND_EYE_METHODS for name in methods):
        known = ", ".join([*HAND_EYE_METHODS, "ALL"])
        raise ValueError(
            f"Unknown hand-eye method {method!r}; expected one of: {known}"
        )

    candidates = []
    errors = []
    for name in methods:
        try:
            candidate = _solve_one_method(T_base_tcp, T_camera_target, name)
            if not np.isfinite(candidate["T_base_camera"]).all():
                raise ValueError("non-finite transform")
            candidates.append(candidate)
        except Exception as exc:  # OpenCV may reject weak pose sets for some methods.
            errors.append({"method": name, "error": str(exc)})

    if not candidates:
        raise RuntimeError(f"All hand-eye methods failed: {errors}")

    best = min(
        candidates,
        key=lambda c: (
            c["summary"]["median_translation_mm"],
            c["summary"]["median_rotation_deg"],
        ),
    )
    best["candidates"] = candidates
    best["failed_candidates"] = errors
    return best
