from __future__ import annotations
from scipy.spatial.transform import Rotation as R
import numpy as np


def construct_adjoint_matrix(tcp_pose):
    """
    Construct the adjoint matrix for a spatial velocity vector
    Used for the robot controlled by twist.
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_transform_matrix(tcp_pose):
    """
    Construct the transform matrix from given pose.
    Used for the robot controlled by pose like provided franka delta pose controller.
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    transform_matrix = np.zeros((6, 6))
    transform_matrix[:3, :3] = rotation
    transform_matrix[3:, 3:] = rotation
    return transform_matrix


def construct_homogeneous_matrix(tcp_pose):
    """
    Construct the homogeneous transformation matrix from given pose.
    args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T


def construct_adjoint_matrix_from_euler(tcp_pose):
    """
    Construct the adjoint matrix for a spatial velocity vector
    :args: tcp_pose: (x, y, z, rx, ry, rz)   where rx,ry,rz are Euler xyz
    """
    rotation = R.from_euler("xyz", tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_homogeneous_matrix_from_euler(tcp_pose):
    """
    Construct the homogeneous transformation matrix from given pose.
    args: tcp_pose: (x, y, z, rx, ry, rz)   where rx,ry,rz are Euler xyz
    """
    rotation = R.from_euler("xyz", tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T


# -----------------------------
# Two small additions for ArUco
# -----------------------------


def construct_homogeneous_matrix_from_rvec_tvec(
    rvec: np.ndarray, tvec: np.ndarray
) -> np.ndarray:
    """
    OpenCV ArUco pose returns rvec (Rodrigues rotvec) and tvec in camera frame.
    Returns a 4x4 homogeneous matrix.
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_rotvec(rvec).as_matrix()
    T[:3, 3] = tvec
    return T


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Applies 4x4 homogeneous T to a 3D point p."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    p = np.asarray(p, dtype=np.float64).reshape(3)
    ph = np.concatenate([p, [1.0]])
    out = T @ ph
    return out[:3]
