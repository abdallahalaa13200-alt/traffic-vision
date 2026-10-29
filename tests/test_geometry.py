import numpy as np
import pytest

from traffic_vision.geometry import (
    apply_homography,
    homography_from_points,
    point_in_polygon,
    segments_intersect,
    side_of_line,
)

LEFT_TO_RIGHT = ((0.0, 50.0), (100.0, 50.0))


def test_side_of_line_sign_in_image_coordinates():
    start, end = LEFT_TO_RIGHT
    assert side_of_line((50, 80), start, end) > 0  # below the line (y grows downward)
    assert side_of_line((50, 20), start, end) < 0  # above
    assert side_of_line((50, 50), start, end) == 0


@pytest.mark.parametrize(
    ("p1", "p2", "expected"),
    [
        ((50, 60), (50, 40), True),  # straight across
        ((10, 70), (90, 30), True),  # diagonal across
        ((50, 60), (50, 55), False),  # stays below
        ((150, 60), (150, 40), False),  # crosses the infinite line, misses the segment
        ((100, 60), (100, 50), True),  # touches the segment's end point
        ((20, 50), (80, 50), True),  # collinear and overlapping
        ((120, 50), (180, 50), False),  # collinear but disjoint
    ],
)
def test_segments_intersect(p1, p2, expected):
    assert segments_intersect(p1, p2, *LEFT_TO_RIGHT) is expected


def test_point_in_polygon():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), square)
    assert not point_in_polygon((15, 5), square)
    assert not point_in_polygon((5, -1), square)

    trapezoid = [(40, 0), (60, 0), (100, 100), (0, 100)]
    assert point_in_polygon((10, 95), trapezoid)
    assert not point_in_polygon((10, 5), trapezoid)


def test_homography_maps_the_four_correspondences():
    src = [(30, 10), (70, 10), (95, 90), (5, 90)]
    dst = [(0, 0), (3.5, 0), (3.5, 20), (0, 20)]
    h = homography_from_points(src, dst)
    np.testing.assert_allclose(apply_homography(h, src), dst, atol=1e-9)


def test_homography_of_a_pure_scale_is_affine():
    src = [(0, 0), (100, 0), (100, 50), (0, 50)]
    dst = [(0, 0), (10, 0), (10, 5), (0, 5)]
    h = homography_from_points(src, dst)
    np.testing.assert_allclose(apply_homography(h, [(50, 25), (20, 40)]), [(5, 2.5), (2, 4)])


def test_homography_rejects_degenerate_points():
    with pytest.raises(ValueError, match="degenerate"):
        homography_from_points([(0, 0), (1, 1), (2, 2), (3, 3)], [(0, 0), (1, 0), (1, 1), (0, 1)])
    with pytest.raises(ValueError, match="four"):
        homography_from_points([(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1)])
