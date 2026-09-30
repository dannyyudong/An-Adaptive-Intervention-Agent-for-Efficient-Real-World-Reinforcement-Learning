#!/usr/bin/env python3
"""Interactively choose RealSense image crops for UR HIL-SERL tasks."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _add_repo_paths() -> None:
    root = _repo_root()
    for path in (root, root / "serl_robot_infra", root / "examples"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def list_realsense_devices() -> list[dict[str, str]]:
    devices = []
    for dev in rs.context().devices:
        devices.append(
            {
                "serial_number": dev.get_info(rs.camera_info.serial_number),
                "name": dev.get_info(rs.camera_info.name),
            }
        )
    return devices


def normalize_camera_spec(name: str, spec: Any) -> dict[str, Any]:
    if isinstance(spec, str):
        return {"name": name, "serial_number": spec, "dim": (640, 480), "fps": 15}

    normalized = dict(spec)
    if "serial" in normalized and "serial_number" not in normalized:
        normalized["serial_number"] = normalized.pop("serial")
    normalized.setdefault("dim", (640, 480))
    normalized.setdefault("fps", 15)
    normalized["name"] = name
    normalized["dim"] = tuple(normalized["dim"])
    return normalized


def parse_camera_arg(value: str) -> dict[str, Any]:
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            "--camera must be NAME:SERIAL or NAME:SERIAL:WIDTHxHEIGHT"
        )

    name, serial = parts[:2]
    spec: dict[str, Any] = {
        "name": name,
        "serial_number": serial,
        "dim": (640, 480),
        "fps": 15,
    }
    if len(parts) == 3:
        width, height = parts[2].lower().split("x", maxsplit=1)
        spec["dim"] = (int(width), int(height))
    return spec


def load_experiment_cameras(experiment: str) -> list[dict[str, Any]]:
    _add_repo_paths()
    module = importlib.import_module(f"experiments.{experiment}.config")
    env_config = module.EnvConfig
    cameras = getattr(env_config, "REALSENSE_CAMERAS")
    return [normalize_camera_spec(name, spec) for name, spec in cameras.items()]


def open_capture(spec: dict[str, Any]):
    from ur_env.camera.rs_capture import RSCapture
    from ur_env.camera.video_capture import VideoCapture

    kwargs = {
        "name": spec["name"],
        "serial_number": spec["serial_number"],
        "dim": tuple(spec.get("dim", (640, 480))),
        "fps": int(spec.get("fps", 15)),
        "rgb": True,
        "depth": False,
        "pointcloud": False,
    }
    return VideoCapture(RSCapture(**kwargs))


def overlay_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    display = frame.copy()
    for idx, line in enumerate(lines):
        y = 28 + idx * 24
        cv2.putText(
            display,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return display


def choose_crop(name: str, cap, warmup_frames: int) -> list[int]:
    window = f"crop: {name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame = None
    for _ in range(max(1, warmup_frames)):
        frame = cap.read()
    assert frame is not None

    h, w = frame.shape[:2]
    crop = [0, h, 0, w]

    print(f"\n[{name}] keys: c/select crop, f/full frame, a/accept, r/reset, q/quit")
    while True:
        frame = cap.read()
        h, w = frame.shape[:2]

        display = frame.copy()
        y0, y1, x0, x1 = crop
        cv2.rectangle(display, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
        display = overlay_text(
            display,
            [
                f"{name}  frame={w}x{h}  crop=[{y0}, {y1}, {x0}, {x1}]",
                "c: select ROI   f: full frame   a/Enter: accept   r: reset   q/Esc: quit",
            ],
        )
        cv2.imshow(window, display)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("c"), ord("s")):
            roi = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
            x, y, roi_w, roi_h = [int(v) for v in roi]
            if roi_w > 0 and roi_h > 0:
                crop = [y, y + roi_h, x, x + roi_w]
                print(f"[{name}] selected crop: {crop}")
        elif key == ord("f"):
            crop = [0, h, 0, w]
            print(f"[{name}] using full frame: {crop}")
        elif key == ord("r"):
            crop = [0, h, 0, w]
            print(f"[{name}] reset to full frame")
        elif key in (ord("a"), 13, 10):
            cv2.destroyWindow(window)
            return crop
        elif key in (ord("q"), 27):
            cv2.destroyWindow(window)
            raise KeyboardInterrupt


def format_python_snippet(crops: dict[str, list[int]]) -> str:
    lines = ["IMAGE_CROP = {"]
    for name, crop in crops.items():
        lines.append(f'    "{name}": {crop},')
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    _add_repo_paths()

    parser = argparse.ArgumentParser(
        description="Open RealSense camera previews and interactively select crop ranges."
    )
    parser.add_argument(
        "--experiment",
        default="ram_insertion",
        help="Experiment config under examples/experiments to read cameras from.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        type=parse_camera_arg,
        help="Override cameras with NAME:SERIAL or NAME:SERIAL:WIDTHxHEIGHT. Can be repeated.",
    )
    parser.add_argument("--only", nargs="*", help="Only crop these camera names.")
    parser.add_argument("--fps", type=int, help="Override capture FPS for all cameras.")
    parser.add_argument(
        "--width", type=int, help="Override capture width for all cameras."
    )
    parser.add_argument(
        "--height", type=int, help="Override capture height for all cameras."
    )
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument(
        "--list", action="store_true", help="List connected RealSense devices and exit."
    )
    parser.add_argument(
        "--output", type=Path, help="Optional JSON output path for crop ranges."
    )
    args = parser.parse_args()

    devices = list_realsense_devices()
    print("Connected RealSense devices:")
    if devices:
        for dev in devices:
            print(f'  {dev["serial_number"]}  {dev["name"]}')
    else:
        print("  none")
    if args.list:
        return

    camera_specs = (
        args.camera if args.camera else load_experiment_cameras(args.experiment)
    )
    if args.only:
        allowed = set(args.only)
        camera_specs = [spec for spec in camera_specs if spec["name"] in allowed]

    for spec in camera_specs:
        if args.width is not None or args.height is not None:
            width, height = spec.get("dim", (640, 480))
            spec["dim"] = (args.width or width, args.height or height)
        if args.fps is not None:
            spec["fps"] = args.fps

    connected_serials = {dev["serial_number"] for dev in devices}
    missing = [
        spec for spec in camera_specs if spec["serial_number"] not in connected_serials
    ]
    if missing:
        missing_text = ", ".join(
            f'{spec["name"]}:{spec["serial_number"]}' for spec in missing
        )
        raise RuntimeError(f"Requested camera(s) not connected: {missing_text}")

    crops: dict[str, list[int]] = {}
    captures = []
    try:
        for spec in camera_specs:
            print(
                f'Opening {spec["name"]}: serial={spec["serial_number"]}, '
                f'dim={tuple(spec.get("dim", (640, 480)))}, fps={int(spec.get("fps", 15))}'
            )
            cap = open_capture(spec)
            captures.append(cap)
            crops[spec["name"]] = choose_crop(spec["name"], cap, args.warmup_frames)
    finally:
        for cap in captures:
            try:
                cap.close()
            except Exception as exc:
                print(f"Failed to close camera cleanly: {exc}")
        cv2.destroyAllWindows()

    print("\nPaste this into EnvConfig:")
    print(format_python_snippet(crops))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(crops, indent=2) + "\n")
        print(f"\nWrote crop JSON to {args.output}")


if __name__ == "__main__":
    main()
