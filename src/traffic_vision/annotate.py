"""Frame overlays: boxes, motion trails, the counting line and a live tally."""

from __future__ import annotations

import colorsys
import zlib
from collections import deque
from collections.abc import Sequence

import cv2
import numpy as np

from .counter import IN, OUT, LineCounter
from .detector import TrackedObject
from .geometry import Point

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_IDLE = (0, 215, 255)  # BGR amber
LINE_HIT = (80, 220, 60)  # BGR green, shown briefly after a crossing
REGION_COLOR = (255, 190, 0)


def class_color(name: str) -> tuple[int, int, int]:
    """Stable BGR colour per class name (crc32, unlike hash(), is the same every run)."""
    hue = (zlib.crc32(name.encode()) % 360) / 360
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _to_int(p: Point) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _put_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    (tw, th), base = cv2.getTextSize(text, FONT, 0.5, 1)
    y = max(y, th + base + 4)
    cv2.rectangle(frame, (x, y - th - base - 4), (x + tw + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - base - 2), FONT, 0.5, (20, 20, 20), 1, cv2.LINE_AA)


class Annotator:
    def __init__(
        self,
        line: tuple[Point, Point],
        trail_length: int = 30,
        region: Sequence[Point] | None = None,
        max_age: int = 30,
    ) -> None:
        self.line = (_to_int(line[0]), _to_int(line[1]))
        self.trail_length = trail_length
        self.region = None if region is None else np.array([_to_int(p) for p in region], np.int32)
        self.max_age = max_age
        self._trails: dict[int, deque[tuple[int, int]]] = {}
        self._last_seen: dict[int, int] = {}

    def draw(
        self,
        frame: np.ndarray,
        objects: Sequence[TrackedObject],
        counter: LineCounter,
        speeds: dict[int, float],
        frame_idx: int,
        highlight: bool = False,
    ) -> np.ndarray:
        """Draws onto ``frame`` in place and returns it."""
        if self.region is not None:
            cv2.polylines(frame, [self.region], True, REGION_COLOR, 1, cv2.LINE_AA)
        cv2.line(frame, *self.line, LINE_HIT if highlight else LINE_IDLE, 3, cv2.LINE_AA)

        for obj in objects:
            color = class_color(obj.class_name)
            self._draw_trail(frame, obj, color, frame_idx)
            x1, y1, x2, y2 = (int(v) for v in obj.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{obj.track_id} {obj.class_name}"
            if obj.track_id in speeds:
                label += f" {speeds[obj.track_id]:.0f} km/h"
            _put_label(frame, label, x1, y1, color)

        self._forget_stale(frame_idx)
        self._draw_panel(frame, counter)
        return frame

    def _draw_trail(
        self, frame: np.ndarray, obj: TrackedObject, color: tuple[int, int, int], frame_idx: int
    ) -> None:
        if self.trail_length == 0:
            return
        trail = self._trails.setdefault(obj.track_id, deque(maxlen=self.trail_length))
        trail.append(_to_int(obj.anchor))
        self._last_seen[obj.track_id] = frame_idx
        if len(trail) > 1:
            cv2.polylines(frame, [np.array(trail, np.int32)], False, color, 2, cv2.LINE_AA)

    def _forget_stale(self, frame_idx: int) -> None:
        stale = [tid for tid, seen in self._last_seen.items() if frame_idx - seen > self.max_age]
        for tid in stale:
            del self._last_seen[tid]
            self._trails.pop(tid, None)

    @staticmethod
    def _draw_panel(frame: np.ndarray, counter: LineCounter) -> None:
        rows = [f"IN  {counter.total_in}", f"OUT {counter.total_out}"]
        for name in sorted(set(counter.counts[IN]) | set(counter.counts[OUT])):
            rows.append(f"{name}: {counter.counts[IN][name]} in / {counter.counts[OUT][name]} out")

        h, w = frame.shape[:2]
        x2, y2 = min(w, 330), min(h, 22 + 26 * len(rows))
        frame[10:y2, 10:x2] = (frame[10:y2, 10:x2] * 0.35).astype(np.uint8)  # dark backdrop
        for i, text in enumerate(rows):
            scale, thickness = (0.75, 2) if i < 2 else (0.55, 1)
            cv2.putText(
                frame, text, (22, 38 + 26 * i), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA
            )
