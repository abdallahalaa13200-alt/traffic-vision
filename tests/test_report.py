import csv
import json

from traffic_vision.counter import IN, OUT, CrossingEvent, LineCounter
from traffic_vision.report import summarize, write_events_csv, write_json


def test_summarize_groups_counts_by_class():
    counter = LineCounter((0.0, 50.0), (100.0, 50.0))
    for tid, cls, path in [
        (1, "car", [(10, 70), (10, 30)]),
        (2, "car", [(20, 30), (20, 70)]),
        (3, "bus", [(30, 70), (30, 30)]),
    ]:
        for frame, p in enumerate(path):
            counter.update(tid, cls, p, frame)

    summary = summarize(counter, source="clip.mp4", frames=300, fps=30.0)
    assert summary["totals"] == {IN: 2, OUT: 1, "all": 3}
    assert summary["by_class"] == {"bus": {IN: 1, OUT: 0}, "car": {IN: 1, OUT: 1}}
    assert summary["duration_s"] == 10.0


def test_events_csv_round_trip(tmp_path):
    events = [
        CrossingEvent(4, "car", IN, 30, speed_kmh=52.345),
        CrossingEvent(9, "truck", OUT, 75),
    ]
    path = tmp_path / "out" / "events.csv"
    write_events_csv(events, path, fps=25.0)

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0] == {
        "frame": "30",
        "time_s": "1.20",
        "track_id": "4",
        "class": "car",
        "direction": IN,
        "speed_kmh": "52.3",
    }
    assert rows[1]["speed_kmh"] == ""


def test_write_json_creates_parent_dirs(tmp_path):
    path = tmp_path / "a" / "b" / "summary.json"
    write_json({"totals": {"all": 3}}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"totals": {"all": 3}}
