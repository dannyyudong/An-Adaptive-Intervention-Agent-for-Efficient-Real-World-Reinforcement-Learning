from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Optional, Dict, Any


class Robotiq2F85USBGripper:
    """
    Async wrapper around pyrobotiqgripper (Modbus RTU over serial) that matches the
    method names used by your existing VacuumGripper.

    IMPORTANT DESIGN CHOICE:
      - We avoid pyrobotiqgripper.goTo() because it blocks until motion is finished.
      - Instead, we send the same Modbus "write_registers(1000, [...])" command
        (non-blocking) and separately poll readAll()/paramDic for state.

    This makes it safe to call from a high-rate robot control loop.
    """

    # Keep same enum names as your VacuumGripper so your controller code can reuse
    class GripperStatus(Enum):
        RESET = 0
        ACTIVATING = 1
        ACTIVE = 3

    class ObjectStatus(Enum):
        MOVING = 0
        DETECTED_MIN = 1
        DETECTED_MAX = 2
        NO_OBJ_DETECTED = 3

    def __init__(
        self,
        portname: str = "auto",
        slaveaddress: int = 9,
        default_speed: int = 255,
        default_force: int = 200,
        cache_period_s: float = 0.02,
        emulate_vacuum_pressure: bool = False,
    ) -> None:
        """
        Args:
            portname: e.g. "/dev/ttyUSB0" or "COM4" or "auto"
            slaveaddress: usually 9 for Robotiq 2F grippers
            default_speed/default_force: [0..255]
            cache_period_s: cache readAll results to avoid double reads per tick
            emulate_vacuum_pressure:
                - True: get_current_pressure() returns a *pseudo* "pressure" (0 or 98)
                        so your existing vacuum-gripper logic doesn't instantly break.
                - False: get_current_pressure() returns gPO (0..255) = actual gripper position.
        """
        # Shutdown guard: once True, we never schedule new to_thread() work.
        self._closing = False

        # Sentinel used by _call_blocking_safe so we can distinguish shutdown
        # from legitimate None returns of blocking functions.
        self._SHUTDOWN = object()

        self.portname = portname
        self.slaveaddress = int(slaveaddress)

        self.default_speed = int(default_speed)
        self.default_force = int(default_force)
        self.cache_period_s = float(cache_period_s)
        self.emulate_vacuum_pressure = bool(emulate_vacuum_pressure)

        self._gripper = None  # pyrobotiqgripper.RobotiqGripper instance
        self._lock = asyncio.Lock()

        # cache of the last readAll()
        self._cache_time = 0.0
        self._cache: Optional[Dict[str, Any]] = None

        # track whether we’ve activated once
        self._activated_once = False

    # -------------------------
    # Internal helpers
    # -------------------------
    @staticmethod
    def _clamp_u8(x: int) -> int:
        return max(0, min(int(x), 255))

    async def _call_blocking(self, fn, *args, **kwargs):
        """
        Run blocking I/O (serial Modbus) off-thread.
        """
        if self._closing:
            return self._SHUTDOWN

        # If there's no running loop or it's closing, do not schedule work.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self._SHUTDOWN

        if loop.is_closed():
            return self._SHUTDOWN

        try:
            return await asyncio.to_thread(fn, *args, **kwargs)

        except asyncio.CancelledError:
            # Task cancelled during shutdown — treat as expected.
            return self._SHUTDOWN

        except RuntimeError as e:
            msg = str(e).lower()

            # Common shutdown errors from executor teardown / interpreter shutdown
            if "cannot schedule new futures" in msg and "shutdown" in msg:
                return self._SHUTDOWN
            if "event loop is closed" in msg:
                return self._SHUTDOWN

            # Anything else is a real error
            raise

    def _require_connected(self):
        if self._gripper is None:
            raise RuntimeError(
                "Robotiq gripper not connected. Call await connect() first."
            )

    async def _read_all_cached(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Calls gripper.readAll() (blocking) and returns a COPY of paramDic.
        Uses a short cache window so get_current_pressure() and get_object_status()
        in the same tick don't double-read the serial line.
        """
        self._require_connected()

        now = time.monotonic()
        if (
            (not force_refresh)
            and (self._cache is not None)
            and ((now - self._cache_time) < self.cache_period_s)
        ):
            return dict(self._cache)

        async with self._lock:
            # re-check inside lock
            now = time.monotonic()
            if (
                (not force_refresh)
                and (self._cache is not None)
                and ((now - self._cache_time) < self.cache_period_s)
            ):
                return dict(self._cache)

            res = await self._call_blocking(self._gripper.readAll)
            if res is self._SHUTDOWN:
                # During shutdown: return best-effort cache or empty dict
                return dict(self._cache) if self._cache is not None else {}

            data = dict(getattr(self._gripper, "paramDic", {}))
            self._cache = data
            self._cache_time = now
            return dict(data)

    async def _write_goto_nonblocking(
        self, position: int, speed: Optional[int] = None, force: Optional[int] = None
    ):
        """
        Send a single Modbus 'go to' command WITHOUT waiting for motion completion.

        This mirrors what pyrobotiqgripper.goTo() sends internally via:
            write_registers(1000, [0b0000100100000000, position, speed*256 + force])
        but skips the busy-wait loop.
        """
        self._require_connected()

        pos = self._clamp_u8(position)
        spd = self._clamp_u8(self.default_speed if speed is None else speed)
        frc = self._clamp_u8(self.default_force if force is None else force)

        reg0 = 0b0000100100000000  # rACT=1 and rGTO=1 encoded as in pyrobotiqgripper
        reg1 = pos
        reg2 = spd * 256 + frc

        async with self._lock:
            res = await self._call_blocking(
                self._gripper.write_registers, 1000, [reg0, reg1, reg2]
            )
            if res is self._SHUTDOWN:
                return

        # Invalidate cache so next read sees fresh state
        self._cache = None

    async def _ensure_active(self):
        if not await self.is_active():
            await self.activate()

    # -------------------------
    # Public API (VacuumGripper-compatible)
    # -------------------------
    async def connect(self) -> None:
        """
        Create the underlying pyrobotiqgripper.RobotiqGripper instance.

        pyrobotiqgripper constructor opens serial and performs an initial readAll().
        """
        # Import here so your program can still start even if gripper not needed
        try:
            from pyrobotiqgripper import RobotiqGripper
        except Exception as e:
            raise ImportError(
                "pyrobotiqgripper is not installed or failed to import. "
                "Install it with: pip install pyrobotiqgripper"
            ) from e

        async with self._lock:
            # Create instrument (blocking)
            g = await self._call_blocking(
                RobotiqGripper, self.portname, self.slaveaddress
            )
            if g is self._SHUTDOWN:
                return
            self._gripper = g

            self._cache = None
            self._activated_once = False

    async def disconnect(self) -> None:
        """
        Close the serial port if available.
        Safe to call during shutdown.
        """
        # First: mark closing so no further to_thread calls get scheduled
        self._closing = True

        if self._gripper is None:
            return

        async with self._lock:
            ser = getattr(self._gripper, "serial", None)
            try:
                if ser is not None:
                    res = await self._call_blocking(ser.close)
                    # ignore shutdown sentinel
            finally:
                self._gripper = None
                self._cache = None
                self._activated_once = False

    async def activate(self) -> None:
        """
        Activates gripper (may take time: it can do motions during activation).
        Only run when needed (startup / fault recovery), not every grip command.
        """
        self._require_connected()
        async with self._lock:
            await self._call_blocking(self._gripper.activate)
        self._activated_once = True
        self._cache = None

    async def is_active(self) -> bool:
        """
        Returns True if gSTA indicates activation completed.
        """
        data = await self._read_all_cached(force_refresh=False)
        gsta = int(data.get("gSTA", 0))
        return gsta == int(self.GripperStatus.ACTIVE.value)

    async def get_fault_status(self) -> int:
        """
        Return gFLT (0 means no fault).
        """
        data = await self._read_all_cached(force_refresh=False)
        return int(data.get("gFLT", 0))

    async def get_object_status(self) -> ObjectStatus:
        """
        Map gOBJ -> VacuumGripper-like ObjectStatus enum.
        """
        data = await self._read_all_cached(force_refresh=False)
        gobj = int(data.get("gOBJ", 0))
        gobj = max(0, min(gobj, 3))
        return self.ObjectStatus(gobj)

    async def get_current_pressure(self) -> int:
        """
        VacuumGripper returns pressure; Robotiq doesn't have pressure.

        Two modes:
          - emulate_vacuum_pressure=True:
              return 98 if object detected (gOBJ in {1,2}), else 0
              (this is the closest drop-in for your current controller logic)
          - emulate_vacuum_pressure=False:
              return gPO (0..255) = actual gripper position from encoders
        """
        data = await self._read_all_cached(force_refresh=False)

        if not self.emulate_vacuum_pressure:
            return int(data.get("gPO", 0))

        gobj = int(data.get("gOBJ", 0))
        if gobj in (1, 2):
            return 98  # mimic "good suction pressure"
        return 0

    # ---- Commands (non-blocking) ----
    async def automatic_grip(self) -> bool:
        """
        Close the gripper (non-blocking command).
        Returns True if the command was sent (not whether object is grasped yet).
        """
        await self._ensure_active()
        await self._write_goto_nonblocking(position=255)
        return True

    async def automatic_release(self) -> bool:
        """
        Open the gripper (non-blocking command).
        """
        await self._ensure_active()
        await self._write_goto_nonblocking(position=0)
        return True

    async def advanced_grip(
        self, min_pressure: int, max_pressure: int, timeout: int
    ) -> bool:
        """
        Vacuum's advanced_grip parameters don't map 1:1 to Robotiq.

        We map:
          - max_pressure -> force (0..255)
          - timeout      -> speed (0..255)  (heuristic)
        min_pressure is ignored (no direct analog).

        Sends a non-blocking close command.
        """
        await self._ensure_active()
        force = self._clamp_u8(max_pressure)
        speed = self._clamp_u8(timeout)
        await self._write_goto_nonblocking(position=255, speed=speed, force=force)
        return True

    async def continuous_grip(self, timeout: int) -> bool:
        """
        Keep commanding a close with some speed. Non-blocking.
        """
        await self._ensure_active()
        speed = self._clamp_u8(timeout)
        await self._write_goto_nonblocking(
            position=255, speed=speed, force=self.default_force
        )
        return True

    async def advanced_release(
        self, min_pressure: int, max_pressure: int, timeout: int
    ) -> bool:
        """
        Non-blocking open. 'timeout' mapped to speed.
        """
        await self._ensure_active()
        speed = self._clamp_u8(timeout)
        await self._write_goto_nonblocking(
            position=0, speed=speed, force=self.default_force
        )
        return True

    # Convenience (not in VacuumGripper, but handy)
    async def move(
        self, position: int, speed: Optional[int] = None, force: Optional[int] = None
    ) -> bool:
        await self._ensure_active()
        await self._write_goto_nonblocking(position=position, speed=speed, force=force)
        return True
