"""Gym Interface for UR5"""

import time
import threading
import copy
import numpy as np
import gymnasium as gym
import cv2
import queue
import warnings
import requests
import json
import os
from typing import Dict, Tuple
from datetime import datetime
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R
import open3d as o3d

from ur_env.camera.video_capture import VideoCapture
from ur_env.camera.rs_capture import RSCapture

from ur_env.camera.utils import PointCloudFusion, CalibrationTread

# from ur_env.utils.pose_estimation import BoxPoseEstimation

from robot_controllers.ur5_controller import UrImpedanceController


class ImageDisplayer(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [v for k, v in img_array.items() if "full" not in k], axis=0
            )
            cv2.namedWindow("RealSense Cameras", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("RealSense Cameras", 300, 700)
            cv2.imshow("RealSense Cameras", frame)
            cv2.waitKey(1)


class PointCloudDisplayer:
    def __init__(self, render_options_path=None, camera_parameters_path=None):
        self.window = o3d.visualization.Visualizer()
        self.window.create_window(height=400, width=400, visible=True)

        self.pc = o3d.geometry.PointCloud()
        if render_options_path:
            self.window.get_render_option().load_from_json(render_options_path)

        self.param = None
        if camera_parameters_path:
            self.param = o3d.io.read_pinhole_camera_parameters(camera_parameters_path)
        self.ctr = self.window.get_view_control()
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.01, origin=[0, 0, 0]
        )

    def display(self, points):
        self.pc.clear()
        # MASSIVE! speed up if float64 is used, see: https://github.com/isl-org/Open3D/issues/1045
        self.pc.points = o3d.utility.Vector3dVector(points.astype(np.float64) / 1000.0)
        self.window.clear_geometries()
        self.window.add_geometry(self.pc)
        # self.window.add_geometry(self.coord_frame)
        if self.param is not None:
            self.ctr.convert_from_pinhole_camera_parameters(self.param, True)

        self.window.poll_events()
        # self.window.update_renderer()

    def close(self):
        self.window.destroy_window()


##############################################################################


class DefaultEnvConfig:
    """Default configuration for UR5Env. Fill in the values below."""

    RESET_Q = np.zeros((6,))
    RANDOM_RESET = (False,)
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_ROT_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    ABS_POSE_RANGE_LIMITS = np.zeros((2,))
    ACTION_SCALE = np.zeros((3,), dtype=np.float32)

    ROBOT_IP: str = "localhost"
    CONTROLLER_HZ: int = 0
    GRIPPER_TIMEOUT: int = 0  # in milliseconds
    ERROR_DELTA: float = 0.0
    FORCEMODE_DAMPING: float = 0.0
    FORCEMODE_TASK_FRAME = np.zeros(
        6,
    )
    FORCEMODE_SELECTION_VECTOR = np.ones(
        6,
    )
    FORCEMODE_LIMITS = np.zeros(
        6,
    )

    REALSENSE_CAMERAS: Dict = {
        "shoulder": "",
        "wrist": "",
    }
    IMAGE_CROP: Dict = {}


##############################################################################


class UR5Env(gym.Env):
    def __init__(
        self,
        hz: int = 100,
        fake_env=False,
        config=DefaultEnvConfig,
        max_episode_length: int = 200000,
        save_video: bool = False,
        camera_mode: str = "rgb",  # one of (rgb, grey, depth, both(rgb depth), pointcloud, none)
    ):
        self.max_episode_length = max_episode_length
        self.curr_path_length = 0
        self.action_scale = config.ACTION_SCALE

        self.config = config

        self.resetQ = config.RESET_Q
        self.curr_reset_pose = np.zeros((7,), dtype=np.float32)

        self.curr_pos = np.zeros((7,), dtype=np.float32)
        self.curr_vel = np.zeros((6,), dtype=np.float32)
        self.curr_Q = np.zeros((6,), dtype=np.float32)
        self.curr_Qd = np.zeros((6,), dtype=np.float32)
        self.curr_force = np.zeros((3,), dtype=np.float32)
        self.curr_torque = np.zeros((3,), dtype=np.float32)

        # self.pose_estimation_ip = config.POSE_ESTIMATION_IP
        # self.pose_est = config.POSE_ESTIMATION
        # self.pose_estimation_ip = None
        # self. pose_est = None
        # self.WF_rot = config.WF_rot
        # self.WF_rot = None
        self.residual_learning_inference = True
        # self.box_error = config.BOX_ERROR
        # self.low_pass_filter_k = config.LOW_PASS_FILTER

        # boxes
        # self.box_pose_est = BoxPoseEstimation(self.pose_estimation_ip) if config.POSE_ESTIMATION else None
        # self.box_pose_est = None
        # self.goal_pose = np.zeros((3,), dtype=np.float32)
        self.box_position = np.zeros((3,), dtype=np.float32)
        self.box_orientation = np.zeros((3,), dtype=np.float32)
        self.init_box_orientation = np.zeros((3,), dtype=np.float32)
        # self._get_goal_pose()
        # self.rotation_generalization = config.ROTATION_GENERALIZATION

        self.gripper_state = np.zeros((2,), dtype=np.float32)
        self.random_reset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rot_range = config.RANDOM_ROT_RANGE
        self.hz = hz
        np.random.seed(0)  # fix seed for fixed (random) initial rotations

        camera_mode = None if camera_mode.lower() == "none" else camera_mode
        if camera_mode is not None and save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.video_record_camera_key = os.getenv(
            "VIDEO_RECORD_CAMERA_KEY",
            str(getattr(config, "VIDEO_RECORD_CAMERA_KEY", "external")),
        )
        self.video_record_camera_keys = [
            key.strip()
            for key in str(self.video_record_camera_key).split(",")
            if key.strip()
        ] or ["external"]
        self.video_record_raw = (
            os.getenv(
                "VIDEO_RECORD_RAW",
                "1" if bool(getattr(config, "VIDEO_RECORD_RAW", True)) else "0",
            )
            != "0"
        )
        self.video_record_dir = os.getenv(
            "VIDEO_RECORD_DIR", str(getattr(config, "VIDEO_RECORD_DIR", "./videos"))
        )
        self.video_record_fps = int(
            os.getenv("VIDEO_RECORD_FPS", str(getattr(config, "VIDEO_RECORD_FPS", hz)))
        )
        self.recording_frames = {key: [] for key in self.video_record_camera_keys}
        self.camera_mode = camera_mode

        self.cost_infos = {}

        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.xy_range = gym.spaces.Box(
            config.ABS_POSE_RANGE_LIMITS[0],
            config.ABS_POSE_RANGE_LIMITS[1],
            dtype=np.float64,
        )
        self.mrp_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )
        self.last_action = np.zeros(self.action_space.shape)

        image_space_definition = {}
        camera_names = list(config.REALSENSE_CAMERAS.keys())
        if camera_mode in ["rgb", "grey", "both"]:
            channel = 1 if camera_mode == "grey" else 3
            for cam_name in camera_names:
                image_space_definition[cam_name] = gym.spaces.Box(
                    0, 255, shape=(128, 128, channel), dtype=np.uint8
                )

        if camera_mode in ["depth", "both"]:
            for cam_name in camera_names:
                image_space_definition[f"{cam_name}_depth"] = gym.spaces.Box(
                    0, 255, shape=(128, 128, 1), dtype=np.uint8
                )

        if camera_mode in ["pointcloud"]:
            image_space_definition["wrist_pointcloud"] = gym.spaces.Box(
                0, 255, shape=(50, 50, 40), dtype=np.uint8
            )
        if camera_mode is not None and camera_mode not in [
            "rgb",
            "both",
            "depth",
            "pointcloud",
            "grey",
        ]:
            raise NotImplementedError(f"camera mode {camera_mode} not implemented")

        state_space = gym.spaces.Dict(
            {
                "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),  # xyz + quat
                "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "gripper_state": gym.spaces.Box(-1.0, 1.0, shape=(2,)),
                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "action": gym.spaces.Box(-1.0, 1.0, shape=self.action_space.shape),
                "gripper_pose": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
                # "boxes": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                # "trajectory": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                # "goal_pose": gym.spaces.Box(-np.inf, np.inf, shape=(6,))
            }
        )

        obs_space_definition = {"state": state_space}
        if self.camera_mode in ["rgb", "both", "depth", "pointcloud", "grey"]:
            obs_space_definition["images"] = gym.spaces.Dict(image_space_definition)

        self.observation_space = gym.spaces.Dict(obs_space_definition)

        self.cycle_count = 0
        self.controller = None
        self.cap = None

        if fake_env:
            print("[UR5Env] is fake!", fake_env)
            return

        self.controller = UrImpedanceController(
            robot_ip=config.ROBOT_IP,
            frequency=config.CONTROLLER_HZ,
            kp=5000,
            kd=2200,
            config=config,
            verbose=False,
            plot=False,
        )
        self.controller.start()  # start Thread

        if self.camera_mode is not None:
            self.init_cameras(config.REALSENSE_CAMERAS)
            self.img_queue = queue.Queue()
            if self.camera_mode in ["pointcloud"]:
                self.displayer = (
                    PointCloudDisplayer()
                )  # o3d displayer cannot be threaded :/
            else:
                self.displayer = ImageDisplayer(self.img_queue)
                self.displayer.start()
            print("[CAM] Cameras are ready!")

        while not self.controller.is_ready():  # wait for controller
            time.sleep(0.1)
        print("[RIC] Controller has started and is ready!")

        if self.camera_mode in ["pointcloud"]:
            voxel_grid_shape = np.array(
                self.observation_space["images"]["wrist_pointcloud"].shape
            )
            # voxel_grid_shape[-1] *= 8     # do not use compacting for now
            # voxel_grid_shape *= 2
            print(f"pointcloud resolution set to: {voxel_grid_shape}")
            self.pointcloud_fusion = PointCloudFusion(
                angle=30.5,
                x_distance=0.185,
                y_distance=-0.01,
                voxel_grid_shape=voxel_grid_shape,
            )

            # load pre calibrated, else calibrate
            if not self.pointcloud_fusion.load_finetuned():
                # TODO make calibration more robust!
                self.calibration_thread = CalibrationTread(
                    pc_fusion=self.pointcloud_fusion, verbose=True
                )
                self.calibration_thread.start()

                self.calibrate_pointcloud_fusion(visualize=True)

    def clip_safety_box(self, next_pos: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        next_pos[:3] = np.clip(
            next_pos[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        orientation_diff = (
            R.from_quat(next_pos[3:]) * R.from_quat(self.curr_reset_pose[3:]).inv()
        ).as_mrp()
        orientation_diff = np.clip(
            orientation_diff, self.mrp_bounding_box.low, self.mrp_bounding_box.high
        )
        next_pos[3:] = (
            R.from_mrp(orientation_diff) * R.from_quat(self.curr_reset_pose[3:])
        ).as_quat()

        return next_pos

    def get_cost_infos(self, done):
        if not done:
            return self.cost_infos.copy()
        cost_infos = self.cost_infos.copy()
        self.cost_infos = {}
        return cost_infos

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # position
        next_pos = self.curr_pos.copy()
        next_pos[:3] = (
            next_pos[:3] + action[:3] * self.action_scale[0]
        )  # + self.trajectory_dir

        next_pos[3:] = (
            R.from_mrp(action[3:6] * self.action_scale[1] / 4.0)
            * R.from_quat(next_pos[3:])
        ).as_quat()  # c * r  --> applies c after r
        next_pos = self.clip_safety_box(next_pos)

        gripper_action = action[6] * self.action_scale[2]

        self._send_pos_command(next_pos)
        self._send_gripper_command(gripper_action)

        self.curr_path_length += 1

        obs = self._get_obs(action)

        reward = self.compute_reward(obs, action)
        truncated = self._is_truncated()

        done = self.curr_path_length >= self.max_episode_length or truncated

        # if not succeed:
        #     try:
        #         succeed = float(reward) > 0.0
        #     except Exception:
        #         succeed = False
        # if truncated:
        #     succeed = False

        done_for_infos = done or truncated

        info = self.get_cost_infos(done_for_infos)
        # info["succeed"] = succeed
        dt = time.time() - start_time
        to_sleep = max(0, (1.0 / self.hz) - dt)
        if to_sleep == 0:
            warnings.warn(
                f"environment could not be within {self.hz} Hz, took {dt:.4f}s!"
            )
        time.sleep(to_sleep)
        # done = False
        return obs, float(reward), bool(done), bool(truncated), info

    def compute_reward(self, obs, action) -> float:
        return 0.0  # overwrite for each task

    def reached_goal_state(self, obs) -> bool:
        return False  # overwrite for each task

    def go_to_rest(self):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """

        # Perform Carteasian reset
        reset_Q = np.zeros((6))
        if self.resetQ.shape == (1, 6):
            reset_Q[:] = self.resetQ.copy()
        elif self.resetQ.shape[1] == 6 and self.resetQ.shape[0] > 1:
            reset_Q[:] = self.resetQ[0, :].copy()  # make random guess
            self.resetQ[:] = np.roll(self.resetQ, -1, axis=0)  # roll one (not random)
        else:
            raise ValueError(f"invalid resetQ dimension: {self.resetQ.shape}")
        # print(np.rad2deg(reset_Q))
        # input("Enter")
        self._send_reset_command(reset_Q)

        while not self.controller.is_reset():
            time.sleep(0.1)  # wait for the reset operation

        self._update_currpos()
        reset_pose = self.controller.get_target_pos()

        if self.random_reset:  # randomize reset position in xy plane
            reset_shift = np.random.uniform(
                np.negative(self.random_xy_range), self.random_xy_range, (2,)
            )
            reset_pose[:2] += reset_shift

            if self.random_rot_range[0] > 0.0:
                random_rot = np.random.triangular(
                    np.negative(self.random_rot_range),
                    0.0,
                    self.random_rot_range,
                    size=(3,),
                )
            else:
                random_rot = np.zeros((3,))
            reset_pose[3:][:] = (
                R.from_quat(reset_pose[3:]) * R.from_mrp(random_rot)
            ).as_quat()

            self.curr_reset_pose[:] = reset_pose

            self.controller.set_target_pos(
                reset_pose
            )  # random movement after resetting
            time.sleep(0.1)
            while self.controller.is_moving():
                time.sleep(0.1)
            return reset_shift
        else:
            self.curr_reset_pose[:] = reset_pose
            return np.zeros((2,))

    def reset(self, **kwargs):
        self.cycle_count += 1
        if self.save_video:
            self.save_video_recording()
        # input("enter")
        shift = self.go_to_rest()
        self.curr_path_length = 0
        # print(self.last_action.shape)
        obs = self._get_obs(np.zeros_like(self.last_action))
        return obs, {"reset_shift": shift}

    def save_video_recording(self):
        try:
            if isinstance(self.recording_frames, dict):
                frame_groups = self.recording_frames
            else:
                frame_groups = {
                    str(self.video_record_camera_key): self.recording_frames
                }
            if any(len(frames) for frames in frame_groups.values()):
                os.makedirs(self.video_record_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                raw_tag = "raw" if self.video_record_raw else "processed"
                for camera_key, frames in frame_groups.items():
                    if not frames:
                        continue
                    video_path = os.path.join(
                        self.video_record_dir,
                        f"{camera_key}_{raw_tag}_{timestamp}.mp4",
                    )
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.video_record_fps,
                        frames[0].shape[:2][::-1],
                    )
                    for frame in frames:
                        video_writer.write(frame)
                    video_writer.release()
                    print(f"Saved video at {video_path}")
            self.recording_frames = {key: [] for key in self.video_record_camera_keys}
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        """Init both cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        print(name_serial_dict)
        for cam_name, cam_spec in name_serial_dict.items():
            if isinstance(cam_spec, str):
                kwargs = {"serial_number": cam_spec}
            else:
                kwargs = dict(cam_spec)
                if "serial" in kwargs and "serial_number" not in kwargs:
                    kwargs["serial_number"] = kwargs.pop("serial")
            print(f"cam serial: {kwargs.get('serial_number')}")
            rgb = self.camera_mode in ["rgb", "both", "grey"]
            depth = self.camera_mode in ["depth", "both"]
            pointcloud = self.camera_mode in ["pointcloud"]
            cap = VideoCapture(
                RSCapture(
                    name=cam_name, rgb=rgb, depth=depth, pointcloud=pointcloud, **kwargs
                )
            )
            self.cap[cam_name] = cap

    def crop_image(self, name, image) -> np.ndarray:
        """
        Center-crop to a square using the min(image height, image width).
        Works for 848x480 (D405 common), 1280x720, etc.
        """
        crop_spec = getattr(self.config, "IMAGE_CROP", {}).get(name)
        if crop_spec is not None:
            if callable(crop_spec):
                return crop_spec(image)
            y0, y1, x0, x1 = [int(v) for v in crop_spec]
            return image[y0:y1, x0:x1, ...]

        h, w = image.shape[:2]
        s = min(h, w)  # square size

        y0 = (h - s) // 2
        x0 = (w - s) // 2

        return image[y0 : y0 + s, x0 : x0 + s, ...]

    def get_image(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        if self.camera_mode == "pointcloud":
            self.pointcloud_fusion.clear()
        for key, cap in self.cap.items():
            try:
                image = cap.read()
                if self.camera_mode in ["rgb", "both", "grey"]:
                    rgb = image[..., :3].astype(np.uint8)
                    if (
                        self.save_video
                        and self.video_record_raw
                        and key in self.video_record_camera_keys
                    ):
                        self.recording_frames.setdefault(key, []).append(
                            np.ascontiguousarray(rgb.copy())
                        )
                    cropped_rgb = self.crop_image(key, rgb)
                    resized = cv2.resize(
                        cropped_rgb,
                        self.observation_space["images"][key].shape[:2][::-1],
                    )
                    # convert to grayscale here
                    if self.camera_mode == "grey":
                        grey = np.array([0.2989, 0.5870, 0.1140])
                        resized = np.dot(resized, grey)[..., None]
                        resized = resized.astype(np.uint8)
                        display_images[key] = np.repeat(resized, 3, axis=-1)
                    else:
                        display_images[key] = resized

                    images[key] = resized[..., ::-1]
                    display_images[key + "_full"] = cropped_rgb
                    if (
                        self.save_video
                        and not self.video_record_raw
                        and key in self.video_record_camera_keys
                    ):
                        self.recording_frames.setdefault(key, []).append(
                            np.ascontiguousarray(display_images[key].copy())
                        )

                if self.camera_mode in ["depth", "both"]:
                    depth_key = key + "_depth"
                    depth = image[..., -1:]
                    cropped_depth = self.crop_image(key, depth)

                    resized = cv2.resize(
                        cropped_depth,
                        np.array(self.observation_space["images"][depth_key].shape[:2])
                        * 3,
                        # (128 * 3, 128 * 3) image
                    )[..., None]

                    resized = resized.reshape((128, 3, 128, 3, 1)).max(
                        (1, 3)
                    )  # max pool with 3x3

                    images[depth_key] = resized
                    display_images[depth_key] = cv2.applyColorMap(
                        resized, cv2.COLORMAP_JET
                    )
                    display_images[depth_key + "_full"] = cv2.applyColorMap(
                        cropped_depth, cv2.COLORMAP_JET
                    )

                if self.camera_mode in ["pointcloud"]:
                    pointcloud = image
                    self.pointcloud_fusion.append(pointcloud)

            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_image()

        if self.camera_mode in ["pointcloud"]:
            (
                voxel_grid,
                voxel_indices,
            ) = self.pointcloud_fusion.get_pointcloud_representation(voxelize=True)

            # downsample on 2x2x2 grid with sum of points (8 as max)
            # vs = self.observation_space["images"]["wrist_pointcloud"].shape
            # voxel_grid = np.sum(np.reshape(voxel_grid, (vs[0], 2, vs[1], 2, vs[2], 2)), axis=(1, 3, 5))
            images["wrist_pointcloud"] = voxel_grid.astype(np.uint8)

            self.displayer.display(voxel_indices)

        # self.recording_frames.append(
        #     np.concatenate([image for key, image in display_images.items() if "full" in key], axis=0)
        # )
        self.img_queue.put(display_images)

        return images

    def calibrate_pointcloud_fusion(self, save=True, visualize=False, num_samples=20):
        self.reset()
        import open3d as o3d

        assert self.camera_mode in ["pointcloud"]
        print("calibrating pointcloud fusion...")
        # calibrate pc fusion here

        obs, reward, done, truncated, _ = self.step(np.zeros((7,)))
        pc = o3d.geometry.PointCloud()
        fused = self.pointcloud_fusion.fuse_pointclouds(voxelize=False, cropped=False)
        pc.points = o3d.utility.Vector3dVector(fused)
        o3d.visualization.draw_geometries([pc])

        # get samples
        for i in range(num_samples):
            # action = [np.sin(i * np.pi / 10.), np.cos(i * np.pi / 10.), 0., -.3 * np.sin(i * np.pi / 10.),
            #           -.3 * np.cos(i * np.pi / 10.), 0., 0.]
            action = [
                -1.0 if i % 4 < 2 else 1,
                -1.0 if i % 4 in [1, 2] else 1,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ]

            # print(action)
            obs, reward, done, truncated, _ = self.step(np.array(action))
            time.sleep(0.1)

            self.calibration_thread.append_backlog(
                *self.pointcloud_fusion.get_original_pcds()
            )

        # calibrate()
        self.controller.stop()
        # self.controller.join(timeout=2.0)
        time.sleep(1)
        self.calibration_thread.calibrate()

        if save:
            self.pointcloud_fusion.save_finetuned()

        if visualize:
            pc = o3d.geometry.PointCloud()
            for i in range(num_samples):
                pc.clear()
                pcs = self.calibration_thread.pc_backlog[i]
                self.pointcloud_fusion.clear()
                self.pointcloud_fusion.append(pcs[0])
                self.pointcloud_fusion.append(pcs[1])
                fused = self.pointcloud_fusion.fuse_pointclouds(
                    voxelize=False, cropped=False
                )
                pc.points = o3d.utility.Vector3dVector(fused)
                o3d.visualization.draw_geometries([pc])

        self.calibration_thread.join()
        exit("restart the program to use the calibrated values")

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _send_pos_command(self, target_pos: np.ndarray):
        """Internal function to send force command to the robot."""
        self.controller.set_target_pos(target_pos=target_pos)

    def _send_gripper_command(self, gripper_pos: np.ndarray):
        self.controller.set_gripper_pos(gripper_pos)

    def _send_reset_command(self, reset_Q: np.ndarray):
        self.controller.set_reset_Q(reset_Q)

    def _send_taskspace_command(self, target_pos):
        self.controller.set_reset_pose(target_pos)

    def _update_box_pos_estimate(self):
        self.box_position = self.box_pose_est.get_box_position()
        self.box_position = self.WF_rot @ self.box_position

    def _update_box_orientation_estimate(self):
        self.box_orientation = self.box_pose_est.get_box_orientation()
        self.box_orientation = (
            R.from_matrix(self.rotation_generalization)
            * R.from_matrix(self.WF_rot)
            * R.from_rotvec(self.box_orientation)
        ).as_rotvec()

    def _update_box_size_estimate(self):
        self.box_size = self.box_pose_est.get_box_size()

    def _get_goal_pose(self):
        """
        Make sure the goal pose is the correct one before computing the reward.
        """
        self.goal_pose = self.config.GOAL_POSE

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        state = self.controller.get_state()

        self.curr_pos[:] = state["pos"]
        self.curr_vel[:] = state["vel"]
        self.curr_force[:] = state["force"]
        self.curr_torque[:] = state["torque"]
        self.curr_Q[:] = state["Q"]
        self.curr_Qd[:] = state["Qd"]
        self.gripper_state[:] = state["gripper"]

    def _is_truncated(self):
        return self.controller.is_truncated()

    def _get_obs(self, action) -> dict:  # Overwritten by box_placing_env.py
        # get image before state observation, so they match better in time

        images = None
        if self.camera_mode is not None:
            images = self.get_image()

        # if self.pose_est:
        #     self._update_box_pos_estimate()
        # else:
        #     self.box_position = np.array([0.5, 0.5, 0.5])

        self._update_currpos()
        state_observation = {
            "tcp_pose": self.curr_pos,
            "tcp_vel": self.curr_vel,
            "gripper_state": self.gripper_state,
            "tcp_force": self.curr_force,
            "tcp_torque": self.curr_torque,
            "gripper_pose": np.array([self.gripper_state[0]], dtype=np.float32),
        }

        if images is not None:
            return copy.deepcopy(dict(images=images, state=state_observation))
        else:
            return copy.deepcopy(dict(state=state_observation))

    def close(self):
        if self.save_video:
            self.save_video_recording()
        if self.controller:
            self.controller.stop()
            # self.controller.join(timeout=2.0)
        # if self.pose_est:
        #     self.box_pose_est.stop()
        super().close()
