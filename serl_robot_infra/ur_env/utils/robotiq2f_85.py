from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ur_env.utils.gripper import Gripper  # Davinci-Kitchen-GmbH/ur-robotiq-gripper


@dataclass
class Robotiq2F85State:
    pos: int  # 0..255
    pos_norm: float  # 0..1 (pos/255)
    is_open: bool
    is_closed: bool
    object_detected: bool  # heuristic


class Robotiq2F85Gripper:
    """
    Async wrapper that avoids blocking the main 100Hz controller loop.
    """

    class ObjectStatus(Enum):
        MOVING = 0
        DETECTED_MIN = 1
        DETECTED_MAX = 2
        NO_OBJ_DETECTED = 3

    def __init__(
        self,
        robot_ip: str,
        port: int = 63352,
        *,
        open_pos: int = 0,
        close_pos: int = 255,
        speed: int = 255,
        force: int = 150,
        object_thresh: int = 245,  # tweak per gripper
    ):
        self.robot_ip = robot_ip
        self.port = int(port)
        self.open_pos = int(open_pos)
        self.close_pos = int(close_pos)
        self.speed = int(speed)
        self.force = int(force)
        self.object_thresh = int(object_thresh)

        self._g: Optional[Gripper] = None
        self._last_cmd: Optional[int] = None
        self._move_task: Optional[asyncio.Task] = None

    async def connect(self):
        self._g = Gripper(self.robot_ip, self.port)
        await self._g.connect()

    async def activate(self, auto_calibrate: bool = True):
        assert self._g is not None
        await self._g.activate(auto_calibrate=auto_calibrate)

    async def open(self, wait: bool = False):
        await self._move(self.open_pos, wait=wait)

    async def close(self, wait: bool = False):
        await self._move(self.close_pos, wait=wait)

    async def _move(self, pos: int, *, wait: bool):
        assert self._g is not None
        pos = int(max(0, min(255, pos)))

        # Avoid spamming identical commands every tick
        if self._last_cmd == pos:
            return
        self._last_cmd = pos

        coro = self._g.move_and_wait_for_pos(pos, self.speed, self.force)

        if wait:
            await coro
            return

        # Run in background so force loop doesn't stall
        if self._move_task is not None and not self._move_task.done():
            self._move_task.cancel()
        self._move_task = asyncio.create_task(coro)

    async def get_state(self) -> Robotiq2F85State:
        assert self._g is not None

        pos = int(await self._g.get_current_position())
        is_open = bool(await self._g.is_open())
        is_closed = bool(await self._g.is_closed())

        # Basic heuristic:
        # if commanded close but can't reach fully closed => likely object
        object_detected = (
            (self._last_cmd == self.close_pos)
            and (pos < self.object_thresh)
            and (not is_closed)
        )

        return Robotiq2F85State(
            pos=pos,
            pos_norm=float(pos) / 255.0,
            is_open=is_open,
            is_closed=is_closed,
            object_detected=object_detected,
        )

    async def get_current_pressure(self) -> int:
        state = await self.get_state()
        return int(state.pos)

    async def get_object_status(self) -> ObjectStatus:
        state = await self.get_state()
        if state.object_detected:
            return self.ObjectStatus.DETECTED_MIN
        return self.ObjectStatus.NO_OBJ_DETECTED

    async def automatic_grip(self) -> bool:
        await self.close(wait=False)
        return True

    async def automatic_release(self) -> bool:
        await self.open(wait=False)
        return True

    async def move(
        self, position: int, speed: Optional[int] = None, force: Optional[int] = None
    ) -> bool:
        assert self._g is not None
        pos = int(max(0, min(255, position)))
        old_speed, old_force = self.speed, self.force
        if speed is not None:
            self.speed = int(max(0, min(255, speed)))
        if force is not None:
            self.force = int(max(0, min(255, force)))
        try:
            await self._move(pos, wait=False)
        finally:
            self.speed, self.force = old_speed, old_force
        return True

    async def disconnect(self):
        if self._move_task is not None and not self._move_task.done():
            self._move_task.cancel()
        if self._g is not None:
            await self._g.disconnect()
            self._g = None
