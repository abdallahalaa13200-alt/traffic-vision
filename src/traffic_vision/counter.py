"""Line-crossing counter driven by tracker IDs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .geometry import Point, segments_intersect, side_of_line

IN = "in"
OUT = "out"


@dataclass(frozen=True)
class CrossingEvent:
    track_id: int
    class_name: str
    direction: str
    frame_idx: int
    speed_kmh: float | None = None


@dataclass
class _TrackState:
    point: Point
    side: float
    last_seen: int


class LineCounter:
    """Counts tracked objects whose anchor point crosses a line segment.

    Moving from the positive side of the directed line ``start -> end`` to the
    negative side is ``"in"``; the opposite is ``"out"``. For a line drawn left
    to right across the frame, that means moving up the frame is ``"in"``.

    Each track ID is counted at most once, so a vehicle that jitters back and
    forth over the line does not inflate the totals. A side change that happens
    beyond the ends of the segment (going around it) is not a crossing.
    """

    def __init__(self, start: Point, end: Point, max_age: int = 90) -> None:
        if start == end:
            raise ValueError("The counting line needs two distinct points.")
        self.start = start
        self.end = end
        self.max_age = max_age
        self.counts: dict[str, Counter[str]] = {IN: Counter(), OUT: Counter()}
        self._tracks: dict[int, _TrackState] = {}
        self._counted: set[int] = set()

    @property
    def total_in(self) -> int:
        return sum(self.counts[IN].values())

    @property
    def total_out(self) -> int:
        return sum(self.counts[OUT].values())

    def update(
        self, track_id: int, class_name: str, point: Point, frame_idx: int
    ) -> CrossingEvent | None:
        """Feed one observation; returns an event if this observation completes a crossing."""
        side = side_of_line(point, self.start, self.end)
        previous = self._tracks.get(track_id)

        if side == 0:
            # Exactly on the line: wait until the object commits to a side.
            if previous is not None:
                previous.last_seen = frame_idx
            return None

        self._tracks[track_id] = _TrackState(point, side, frame_idx)
        if previous is None or track_id in self._counted or (previous.side > 0) == (side > 0):
            return None
        if not segments_intersect(previous.point, point, self.start, self.end):
            return None

        direction = IN if previous.side > 0 else OUT
        self._counted.add(track_id)
        self.counts[direction][class_name] += 1
        return CrossingEvent(track_id, class_name, direction, frame_idx)

    def prune(self, frame_idx: int) -> None:
        """Forget tracks unseen for more than ``max_age`` frames so memory stays bounded."""
        stale = [tid for tid, s in self._tracks.items() if frame_idx - s.last_seen > self.max_age]
        for tid in stale:
            del self._tracks[tid]
