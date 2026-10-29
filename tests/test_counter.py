import pytest

from traffic_vision.counter import IN, OUT, LineCounter


@pytest.fixture
def counter():
    # Horizontal line across a 100-px-wide frame at y = 50, drawn left to right.
    return LineCounter((0.0, 50.0), (100.0, 50.0), max_age=10)


def drive(counter, track_id, points, cls="car", start_frame=0):
    return [
        e
        for i, p in enumerate(points)
        if (e := counter.update(track_id, cls, p, start_frame + i)) is not None
    ]


def test_moving_up_the_frame_counts_as_in(counter):
    events = drive(counter, 1, [(50, 70), (50, 60), (50, 45), (50, 30)])
    assert [(e.track_id, e.direction, e.frame_idx) for e in events] == [(1, IN, 2)]
    assert counter.total_in == 1 and counter.total_out == 0


def test_moving_down_the_frame_counts_as_out(counter):
    events = drive(counter, 7, [(30, 20), (30, 40), (30, 55)], cls="truck")
    assert [e.direction for e in events] == [OUT]
    assert counter.counts[OUT]["truck"] == 1


def test_first_observation_never_counts(counter):
    assert counter.update(1, "car", (50, 40), 0) is None
    assert counter.total_in + counter.total_out == 0


def test_jitter_over_the_line_is_counted_once(counter):
    events = drive(counter, 3, [(50, 52), (50, 48), (50, 52), (50, 48), (50, 52)])
    assert len(events) == 1
    assert counter.total_in == 1 and counter.total_out == 0


def test_landing_exactly_on_the_line_waits_for_a_side(counter):
    events = drive(counter, 4, [(50, 60), (50, 50), (50, 50), (50, 40)])
    assert [(e.direction, e.frame_idx) for e in events] == [(IN, 3)]


def test_passing_beyond_the_segment_end_is_not_a_crossing():
    counter = LineCounter((0.0, 50.0), (60.0, 50.0))
    assert drive(counter, 1, [(80, 70), (80, 30)]) == []


def test_tracks_are_independent_and_counted_per_class(counter):
    drive(counter, 1, [(10, 70), (10, 30)], cls="car")
    drive(counter, 2, [(20, 70), (20, 30)], cls="car")
    drive(counter, 3, [(30, 30), (30, 70)], cls="bus")
    assert dict(counter.counts[IN]) == {"car": 2}
    assert dict(counter.counts[OUT]) == {"bus": 1}


def test_prune_forgets_stale_tracks(counter):
    counter.update(1, "car", (50, 70), frame_idx=0)
    counter.prune(frame_idx=50)
    # With its history gone, the next sighting is a fresh first observation.
    assert counter.update(1, "car", (50, 30), frame_idx=51) is None


def test_prune_keeps_recent_tracks(counter):
    counter.update(1, "car", (50, 70), frame_idx=0)
    counter.prune(frame_idx=5)
    assert counter.update(1, "car", (50, 30), frame_idx=6) is not None


def test_rejects_zero_length_line():
    with pytest.raises(ValueError):
        LineCounter((10.0, 10.0), (10.0, 10.0))
