"""Command-line entry point: ``traffic-vision --config configs/default.yaml --source video.mp4``."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from .config import Config, LineConfig, load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="traffic-vision",
        description="Detect, track, count and time vehicles in a video.",
    )
    p.add_argument("--config", type=Path, help="YAML config (see configs/default.yaml).")
    p.add_argument("--source", help="Video file, stream URL or webcam index.")
    p.add_argument("--output", help="Directory for the annotated video and reports.")
    p.add_argument("--weights", help="YOLO weights, e.g. yolov8s.pt or a custom checkpoint.")
    p.add_argument("--conf", type=float, help="Detection confidence threshold.")
    p.add_argument("--device", help='Inference device: "cpu", "0", "cuda:0", "mps".')
    p.add_argument(
        "--line",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Counting line in normalised [0, 1] coordinates.",
    )
    p.add_argument("--show", action="store_true", help="Display frames while processing.")
    p.add_argument("--no-video", action="store_true", help="Skip writing the annotated video.")
    return p


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """CLI flags win over the YAML file; anything not passed keeps its configured value."""
    if args.source is not None:
        cfg.source = args.source
    if args.output is not None:
        cfg.output_dir = args.output
    if args.weights is not None:
        cfg.model.weights = args.weights
    if args.conf is not None:
        cfg.model.conf = args.conf
    if args.device is not None:
        cfg.model.device = args.device
    if args.line is not None:
        cfg.line = LineConfig(start=tuple(args.line[:2]), end=tuple(args.line[2:]))
    if args.show:
        cfg.show = True
    if args.no_video:
        cfg.save_video = False
    cfg.validate()
    return cfg


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = apply_overrides(load_config(args.config), args)

    from .pipeline import run  # deferred: pulls in OpenCV and the YOLO stack

    summary = run(cfg)
    print(json.dumps(summary, indent=2))
    return 0
