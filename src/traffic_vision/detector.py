"""Thin wrapper around an Ultralytics YOLO model running with a multi-object tracker."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import ModelConfig
from .geometry import Point


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    @property
    def anchor(self) -> Point:
        """Bottom-centre of the box: roughly where the vehicle touches the road."""
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)


class VehicleTracker:
    """Runs detection + tracking on one frame at a time and returns tracked objects."""

    def __init__(self, cfg: ModelConfig, class_names: Sequence[str]) -> None:
        # Imported here so the counting/speed logic stays importable without PyTorch.
        from ultralytics import YOLO

        self._cfg = cfg
        self._model = YOLO(cfg.weights)
        self._names: dict[int, str] = self._model.names
        name_to_id = {name: idx for idx, name in self._names.items()}
        missing = [c for c in class_names if c not in name_to_id]
        if missing:
            raise ValueError(
                f"{cfg.weights} has no class(es) {missing}. Available: {sorted(name_to_id)}"
            )
        self._class_ids = [name_to_id[c] for c in class_names]

    def update(self, frame: np.ndarray) -> list[TrackedObject]:
        result = self._model.track(
            frame,
            persist=True,
            tracker=self._cfg.tracker,
            classes=self._class_ids,
            conf=self._cfg.conf,
            iou=self._cfg.iou,
            imgsz=self._cfg.imgsz,
            device=self._cfg.device,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or boxes.id is None:  # no confirmed tracks in this frame
            return []

        ids = boxes.id.int().cpu().tolist()
        cls = boxes.cls.int().cpu().tolist()
        conf = boxes.conf.cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        return [
            TrackedObject(tid, c, self._names[c], float(p), tuple(float(v) for v in box))
            for tid, c, p, box in zip(ids, cls, conf, xyxy, strict=True)
        ]
