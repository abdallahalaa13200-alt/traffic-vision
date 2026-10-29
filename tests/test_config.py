from pathlib import Path

import pytest

from traffic_vision.cli import apply_overrides, build_parser
from traffic_vision.config import Config, config_from_dict, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_defaults_are_valid():
    Config().validate()


def test_shipped_config_loads():
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert cfg.classes == ["car", "motorcycle", "bus", "truck"]
    assert cfg.model.tracker == "bytetrack.yaml"


def test_yaml_sections_are_merged(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "source: 0\nmodel:\n  weights: yolov8s.pt\n  conf: 0.5\nline:\n  start: [0, 0.3]\n"
        "  end: [1, 0.3]\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.source == "0"
    assert cfg.model.weights == "yolov8s.pt" and cfg.model.conf == 0.5
    assert cfg.model.iou == 0.5  # untouched default
    assert cfg.line.start == (0.0, 0.3)


def test_unknown_keys_are_reported():
    with pytest.raises(ValueError, match="model: confidence"):
        config_from_dict({"model": {"confidence": 0.4}})
    with pytest.raises(ValueError, match="top level: sauce"):
        config_from_dict({"sauce": "video.mp4"})


@pytest.mark.parametrize(
    "raw",
    [
        {"line": {"start": [1.2, 0.5], "end": [0, 0.5]}},
        {"line": {"start": [0.5, 0.5], "end": [0.5, 0.5]}},
        {"model": {"conf": 1.5}},
        {"classes": []},
        {
            "speed": {
                "enabled": True,
                "region": [[0, 0], [1, 0], [1, 1]],
                "width_m": 3,
                "length_m": 9,
            }
        },
        {"speed": {"enabled": True, "region": [[0, 0], [1, 0], [1, 1], [0, 1]], "width_m": 0}},
    ],
)
def test_invalid_configs_are_rejected(raw):
    with pytest.raises(ValueError):
        config_from_dict(raw)


def test_line_and_region_scale_to_pixels():
    cfg = config_from_dict(
        {
            "line": {"start": [0, 0.5], "end": [1, 0.5]},
            "speed": {
                "enabled": True,
                "region": [[0.25, 0.25], [0.75, 0.25], [1, 1], [0, 1]],
                "width_m": 7,
                "length_m": 20,
            },
        }
    )
    assert cfg.line.to_pixels(1920, 1080) == ((0.0, 540.0), (1920.0, 540.0))
    assert cfg.speed.region_pixels(200, 100)[0] == (50.0, 25.0)


def test_cli_flags_override_the_config():
    args = build_parser().parse_args(
        [
            "--source",
            "cam.mp4",
            "--weights",
            "yolo11s.pt",
            "--line",
            "0",
            "0.4",
            "1",
            "0.4",
            "--no-video",
        ]
    )
    cfg = apply_overrides(Config(), args)
    assert cfg.source == "cam.mp4"
    assert cfg.model.weights == "yolo11s.pt"
    assert cfg.line.start == (0.0, 0.4) and cfg.line.end == (1.0, 0.4)
    assert cfg.save_video is False
    assert cfg.model.conf == 0.35  # not passed, so unchanged


def test_cli_rejects_an_invalid_override():
    args = build_parser().parse_args(["--conf", "2"])
    with pytest.raises(ValueError):
        apply_overrides(Config(), args)
