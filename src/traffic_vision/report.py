"""Persist crossing events and run summaries."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .counter import IN, OUT, CrossingEvent, LineCounter


def summarize(counter: LineCounter, *, source: str, frames: int, fps: float) -> dict[str, Any]:
    classes = sorted(set(counter.counts[IN]) | set(counter.counts[OUT]))
    return {
        "source": source,
        "frames_processed": frames,
        "fps": round(fps, 3),
        "duration_s": round(frames / fps, 2) if fps else None,
        "totals": {
            IN: counter.total_in,
            OUT: counter.total_out,
            "all": counter.total_in + counter.total_out,
        },
        "by_class": {
            name: {IN: counter.counts[IN][name], OUT: counter.counts[OUT][name]} for name in classes
        },
    }


def write_events_csv(events: Iterable[CrossingEvent], path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "track_id", "class", "direction", "speed_kmh"])
        for e in events:
            writer.writerow(
                [
                    e.frame_idx,
                    f"{e.frame_idx / fps:.2f}",
                    e.track_id,
                    e.class_name,
                    e.direction,
                    "" if e.speed_kmh is None else f"{e.speed_kmh:.1f}",
                ]
            )


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
