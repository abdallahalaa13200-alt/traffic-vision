"""End-to-end video loop: detect + track -> count -> estimate speed -> draw -> save."""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Any

import cv2

from .annotate import Annotator
from .config import Config
from .counter import CrossingEvent, LineCounter
from .detector import VehicleTracker
from .report import summarize, write_events_csv, write_json
from .speed import SpeedEstimator

log = logging.getLogger(__name__)

FLASH_FRAMES = 8  # how long the line stays green after a crossing
FALLBACK_FPS = 30.0  # webcams and some streams report 0 fps


def _open_source(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")
    return cap


def run(cfg: Config) -> dict[str, Any]:
    cap = _open_source(cfg.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 1.0 <= fps <= 240.0:
        fps = FALLBACK_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        cap.release()
        raise RuntimeError(f"Could not read frame size from {cfg.source}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep per-track state for ~3 s of absence: longer than the tracker's own buffer.
    max_age = int(fps * 3)

    line = cfg.line.to_pixels(width, height)
    tracker = VehicleTracker(cfg.model, cfg.classes)
    counter = LineCounter(*line, max_age=max_age)
    speed: SpeedEstimator | None = None
    region = None
    if cfg.speed.enabled:
        region = cfg.speed.region_pixels(width, height)
        speed = SpeedEstimator(
            region, cfg.speed.width_m, cfg.speed.length_m, fps, cfg.speed.window_s, max_age
        )
    annotator = Annotator(line, cfg.trail_length, region, max_age=int(fps))

    writer = None
    if cfg.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), fourcc, fps, (width, height))

    events: list[CrossingEvent] = []
    frame_idx = 0
    last_event_frame = -FLASH_FRAMES - 1
    started = time.perf_counter()
    log.info("Processing %s (%dx%d @ %.1f fps)", cfg.source, width, height, fps)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            objects = tracker.update(frame)
            for obj in objects:
                kmh = speed.update(obj.track_id, obj.anchor, frame_idx) if speed else None
                event = counter.update(obj.track_id, obj.class_name, obj.anchor, frame_idx)
                if event is not None:
                    event = dataclasses.replace(event, speed_kmh=kmh)
                    events.append(event)
                    last_event_frame = frame_idx
                    log.info(
                        "frame %d: #%d %s %s",
                        frame_idx,
                        event.track_id,
                        event.class_name,
                        event.direction,
                    )

            counter.prune(frame_idx)
            if speed is not None:
                speed.prune(frame_idx)

            annotator.draw(
                frame,
                objects,
                counter,
                speed.speeds if speed else {},
                frame_idx,
                highlight=frame_idx - last_event_frame <= FLASH_FRAMES,
            )
            if writer is not None:
                writer.write(frame)
            if cfg.show:
                cv2.imshow("traffic-vision", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 200 == 0:
                elapsed = time.perf_counter() - started
                log.info("%d frames, %.1f fps", frame_idx, frame_idx / elapsed)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if cfg.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    summary = summarize(counter, source=cfg.source, frames=frame_idx, fps=fps)
    summary["processing_fps"] = round(frame_idx / elapsed, 2) if elapsed > 0 else None
    write_events_csv(events, out_dir / "events.csv", fps)
    write_json(summary, out_dir / "summary.json")
    log.info("Wrote results to %s", out_dir)
    return summary
