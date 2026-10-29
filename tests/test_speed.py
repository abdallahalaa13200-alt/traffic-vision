import pytest

from traffic_vision.speed import SpeedEstimator

# A 100 x 100 px square calibrated as a 10 m x 10 m patch of road: 0.1 m per pixel.
SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def make(fps=10.0, window_s=0.5):
    return SpeedEstimator(SQUARE, width_m=10.0, length_m=10.0, fps=fps, window_s=window_s)


def test_constant_motion_gives_the_expected_speed():
    est = make()
    speed = None
    for frame, y in enumerate(range(5, 96, 10)):  # 10 px/frame = 1 m/frame = 10 m/s at 10 fps
        speed = est.update(1, (50, y), frame)
    assert speed == pytest.approx(36.0)
    assert est.speeds[1] == pytest.approx(36.0)


def test_no_estimate_until_enough_motion_is_observed():
    est = make()
    assert est.update(1, (50, 10), 0) is None
    assert est.update(1, (50, 20), 1) is None
    assert est.update(1, (50, 30), 2) is not None


def test_points_outside_the_region_are_ignored():
    est = make()
    assert est.update(1, (150, 50), 0) is None
    assert 1 not in est.speeds


def test_last_estimate_is_kept_after_leaving_the_region():
    est = make()
    for frame, y in enumerate((10, 20, 30, 40)):
        est.update(1, (50, y), frame)
    inside = est.speeds[1]
    assert est.update(1, (50, 150), 4) == pytest.approx(inside)


def test_prune_drops_stale_tracks():
    est = make()
    for frame, y in enumerate((10, 20, 30)):
        est.update(1, (50, y), frame)
    est.prune(frame_idx=500)
    assert est.speeds == {}


@pytest.mark.parametrize(
    "kwargs",
    [{"fps": 0}, {"width_m": 0}, {"length_m": -1}],
)
def test_invalid_parameters_are_rejected(kwargs):
    params = {"width_m": 10.0, "length_m": 10.0, "fps": 25.0, **kwargs}
    with pytest.raises(ValueError):
        SpeedEstimator(SQUARE, **params)
