"""Ground-plane speed estimation from a four-point perspective calibration."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np

from .geometry import Point, apply_homography, homography_from_points, point_in_polygon


class SpeedEstimator:
    """Estimates speed in km/h by projecting image points onto a metric ground plane.

    ``region`` holds four pixel coordinates outlining a rectangle on the road,
    ordered top-left, top-right, bottom-right, bottom-left as seen in the frame;
    ``width_m`` and ``length_m`` are that rectangle's real dimensions. The
    projection is only trustworthy on the calibrated patch of road, so
    observations outside it are ignored and the last estimate is kept.
    """

    def __init__(
        self,
        region: Sequence[Point],
        width_m: float,
        length_m: float,
        fps: float,
        window_s: float = 0.5,
        max_age: int = 90,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive.")
        if width_m <= 0 or length_m <= 0:
            raise ValueError("width_m and length_m must be positive.")

        self.region = [(float(x), float(y)) for x, y in region]
        world = [(0.0, 0.0), (width_m, 0.0), (width_m, length_m), (0.0, length_m)]
        self._h = homography_from_points(self.region, world)
        self.fps = float(fps)
        self.max_age = max_age
        self._window = max(2, round(window_s * fps))
        self._history: dict[int, deque[tuple[int, np.ndarray]]] = {}
        self._speeds: dict[int, float] = {}

    @property
    def speeds(self) -> dict[int, float]:
        """Latest estimate per track ID."""
        return self._speeds

    def update(self, track_id: int, point: Point, frame_idx: int) -> float | None:
        if not point_in_polygon(point, self.region):
            return self._speeds.get(track_id)

        world = apply_homography(self._h, [point])[0]
        history = self._history.setdefault(track_id, deque(maxlen=self._window))
        history.append((frame_idx, world))

        (first_frame, first_pos), (last_frame, last_pos) = history[0], history[-1]
        # Require at least half a window of motion so one-frame jitter can't produce a speed.
        if last_frame - first_frame < self._window // 2:
            return self._speeds.get(track_id)

        seconds = (last_frame - first_frame) / self.fps
        speed = float(np.linalg.norm(last_pos - first_pos)) / seconds * 3.6
        self._speeds[track_id] = speed
        return speed

    def prune(self, frame_idx: int) -> None:
        stale = [tid for tid, h in self._history.items() if frame_idx - h[-1][0] > self.max_age]
        for tid in stale:
            del self._history[tid]
            self._speeds.pop(tid, None)
