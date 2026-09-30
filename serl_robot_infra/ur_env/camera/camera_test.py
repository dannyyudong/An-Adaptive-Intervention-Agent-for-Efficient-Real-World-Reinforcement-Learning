import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import time
import open3d as o3d

devices = rs.context().devices
print([d.get_info(rs.camera_info.serial_number) for d in devices])
