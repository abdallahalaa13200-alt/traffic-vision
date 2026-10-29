"""Typed run configuration loaded from YAML.

The counting line and the speed-calibration region are stored in normalised
coordinates (fractions of frame width/height) so one config works at any
resolution; they are converted to pixels once the video is opened.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .geometry import Point


@dataclass
class ModelConfig:
    weights: str = "yolov8n.pt"
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    device: str | None = None
    tracker: str = "bytetrack.yaml"


@dataclass
class LineConfig:
    start: tuple[float, float] = (0.05, 0.6)
    end: tuple[float, float] = (0.95, 0.6)

    def __post_init__(self) -> None:
        self.start = tuple(float(v) for v in self.start)
        self.end = tuple(float(v) for v in self.end)

    def to_pixels(self, width: int, height: int) -> tuple[Point, Point]:
        return (
            (self.start[0] * width, self.start[1] * height),
            (self.end[0] * width, self.end[1] * height),
        )


@dataclass
class SpeedConfig:
    enabled: bool = False
    region: list[list[float]] = field(default_factory=list)
    width_m: float = 0.0
    length_m: float = 0.0
    window_s: float = 0.5

    def region_pixels(self, width: int, height: int) -> list[Point]:
        return [(x * width, y * height) for x, y in self.region]


@dataclass
class Config:
    source: str = "0"
    output_dir: str = "runs/latest"
    classes: list[str] = field(default_factory=lambda: ["car", "motorcycle", "bus", "truck"])
    save_video: bool = True
    show: bool = False
    trail_length: int = 30
    model: ModelConfig = field(default_factory=ModelConfig)
    line: LineConfig = field(default_factory=LineConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)

    def validate(self) -> None:
        for name, pt in (("line.start", self.line.start), ("line.end", self.line.end)):
            _check_normalised(name, pt)
        if self.line.start == self.line.end:
            raise ValueError("line.start and line.end must differ.")
        if not 0.0 < self.model.conf < 1.0:
            raise ValueError(f"model.conf must be in (0, 1), got {self.model.conf}.")
        if not 0.0 < self.model.iou < 1.0:
            raise ValueError(f"model.iou must be in (0, 1), got {self.model.iou}.")
        if not self.classes:
            raise ValueError("classes must list at least one class name.")
        if self.trail_length < 0:
            raise ValueError("trail_length cannot be negative.")
        if self.speed.enabled:
            if len(self.speed.region) != 4:
                raise ValueError("speed.region needs exactly four corner points.")
            for i, pt in enumerate(self.speed.region):
                _check_normalised(f"speed.region[{i}]", pt)
            if self.speed.width_m <= 0 or self.speed.length_m <= 0:
                raise ValueError("speed.width_m and speed.length_m must be positive.")
            if self.speed.window_s <= 0:
                raise ValueError("speed.window_s must be positive.")


_SECTIONS = {"model": ModelConfig, "line": LineConfig, "speed": SpeedConfig}


def _check_normalised(name: str, pt: Any) -> None:
    if len(pt) != 2 or not all(0.0 <= float(v) <= 1.0 for v in pt):
        raise ValueError(f"{name} must be a normalised (x, y) pair in [0, 1], got {pt}.")


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) in {where}: {', '.join(unknown)}")
    return cls(**data)


def config_from_dict(raw: dict[str, Any] | None) -> Config:
    raw = dict(raw or {})
    nested = {name: _build(cls, raw.pop(name, None) or {}, name) for name, cls in _SECTIONS.items()}
    if "source" in raw:
        raw["source"] = str(raw["source"])
    cfg = _build(Config, {**raw, **nested}, "top level")
    cfg.validate()
    return cfg


def load_config(path: Path | str | None = None) -> Config:
    if path is None:
        return config_from_dict({})
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return config_from_dict(raw)
