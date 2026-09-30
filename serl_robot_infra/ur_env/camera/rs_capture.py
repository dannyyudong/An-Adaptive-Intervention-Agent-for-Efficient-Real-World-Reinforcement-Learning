import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import time
import open3d as o3d
import threading


class RSCapture:
    def get_device_serial_numbers(self):
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def __init__(
        self,
        name,
        serial_number,
        dim=(640, 480),
        fps=15,
        rgb=True,
        depth=False,
        pointcloud=False,
        exposure=None,
        capture_depth=False,
        cache_raw_rgbd=False,
    ):
        self.name = name
        print(serial_number)
        print(self.get_device_serial_numbers())
        assert serial_number in self.get_device_serial_numbers()
        self.serial_number = serial_number
        self.rgb = rgb
        self.depth = depth
        self.pointcloud = pointcloud
        self.capture_depth = bool(capture_depth)
        self.cache_raw_rgbd = bool(cache_raw_rgbd)
        self._raw_rgbd_lock = threading.Lock()
        self.last_rgb_bgr = None
        self.last_depth_m = None
        self.last_intrinsics = None
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)

        assert self.rgb or self.depth or self.pointcloud
        if self.rgb:
            self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        needs_depth_stream = self.depth or self.capture_depth or self.cache_raw_rgbd
        if needs_depth_stream:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
        if self.pointcloud:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)

        self.profile = self.pipe.start(self.cfg)

        if exposure is not None:
            for sensor in self.profile.get_device().query_sensors():
                if sensor.supports(rs.option.exposure):
                    if sensor.supports(rs.option.enable_auto_exposure):
                        sensor.set_option(rs.option.enable_auto_exposure, 0)
                    sensor.set_option(rs.option.exposure, float(exposure))

        if needs_depth_stream:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            if self.depth:
                self.max_clipping_distance = (
                    0.2 / self.depth_scale
                )  # 0.15m max clipping distance
        else:
            self.depth_scale = None

        if self.pointcloud:
            self.pc = rs.pointcloud()
            self.threshold_filter = rs.threshold_filter(min_dist=0.0, max_dist=0.25)
            self.decimation_filter = rs.decimation_filter(magnitude=2.0)  # 2 or 4
            self.temporal_filter = rs.temporal_filter(
                smooth_alpha=0.53, smooth_delta=24.0, persistence_control=2
            )  # standard values

        # for some weird reason, these values have to be set in order for the image to appear with good lightning
        # for firmware >5.13, auto_exposure False & auto_white_balance True, below only auto_exposure True
        # for sensor in self.profile.get_device().query_sensors():
        #     sensor.set_option(rs.option.enable_auto_exposure, True)
        # sensor.set_option(rs.option.enable_auto_white_balance, True)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def read(self):
        t = time.time()
        frames = self.pipe.wait_for_frames()
        tdiff = time.time() - t
        if tdiff > 0.5:
            print(f"wait for frames took {tdiff:.3f} seconds")
        image, depth, pointcloud = None, None, None
        depth_raw = None
        color_frame = None
        depth_frame = None
        aligned_frames = None

        if self.rgb or self.depth or self.capture_depth or self.cache_raw_rgbd:
            aligned_frames = self.align.process(frames)

        if self.rgb:
            color_frame = aligned_frames.get_color_frame()

            if color_frame.is_video_frame():
                image = np.asarray(color_frame.get_data())

        if self.depth or self.capture_depth or self.cache_raw_rgbd:
            depth_frame = aligned_frames.get_depth_frame()

            if depth_frame.is_depth_frame():
                depth_raw = np.asanyarray(depth_frame.get_data())
                if self.depth:
                    # clip max
                    depth = np.where(
                        (depth_raw > self.max_clipping_distance),
                        0.0,
                        self.max_clipping_distance - depth_raw,
                    )

                    depth = (depth * (256.0 / self.max_clipping_distance)).astype(
                        np.uint8
                    )
                    depth = depth[..., None]

        if (
            self.cache_raw_rgbd
            and isinstance(image, np.ndarray)
            and isinstance(depth_raw, np.ndarray)
        ):
            rs_intr = color_frame.profile.as_video_stream_profile().intrinsics
            intrinsics = {
                "width": int(rs_intr.width),
                "height": int(rs_intr.height),
                "fx": float(rs_intr.fx),
                "fy": float(rs_intr.fy),
                "cx": float(rs_intr.ppx),
                "cy": float(rs_intr.ppy),
            }
            depth_m = depth_raw.astype(np.float32) * float(self.depth_scale)
            with self._raw_rgbd_lock:
                self.last_rgb_bgr = image.copy()
                self.last_depth_m = depth_m.copy()
                self.last_intrinsics = intrinsics

        if self.pointcloud:
            depth_frame = self.decimation_filter.process(frames.get_depth_frame())
            depth_frame = self.threshold_filter.process(depth_frame)
            depth_frame = self.temporal_filter.process(depth_frame)
            if depth_frame.is_depth_frame():
                points = self.pc.calculate(depth_frame)
                pointcloud = (
                    np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
                )

        if isinstance(image, np.ndarray) and isinstance(depth, np.ndarray):
            return True, np.concatenate((image, depth), axis=-1)
        elif isinstance(image, np.ndarray):
            return True, image
        elif isinstance(depth, np.ndarray):
            return True, depth
        elif isinstance(pointcloud, np.ndarray):
            return True, pointcloud
        else:
            return False, None

    def get_raw_rgbd_snapshot(self):
        if not self.cache_raw_rgbd:
            raise RuntimeError(f"{self.name} raw RGB-D cache is disabled")
        with self._raw_rgbd_lock:
            if (
                self.last_rgb_bgr is None
                or self.last_depth_m is None
                or self.last_intrinsics is None
            ):
                raise RuntimeError(f"{self.name} raw RGB-D cache is not ready yet")
            return {
                "rgb": self.last_rgb_bgr[:, :, ::-1].copy(),
                "depth_m": self.last_depth_m.copy(),
                "intrinsics": dict(self.last_intrinsics),
            }

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()
