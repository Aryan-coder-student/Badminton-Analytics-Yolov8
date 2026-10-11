import unittest

import numpy as np

from scratchpad.court_players import (
    is_complete_court_grid,
    is_stable_court_grid_sequence,
)


def create_wide_court_grid() -> np.ndarray:
    grid = np.empty((6, 5, 2), dtype=np.float32)
    for row_index, y_coordinate in enumerate(np.linspace(220, 650, 6)):
        row_width = 480 + row_index * 52
        grid[row_index, :, 0] = np.linspace(
            640 - row_width / 2,
            640 + row_width / 2,
            5,
        )
        grid[row_index, :, 1] = y_coordinate
    return grid


class CourtViewTest(unittest.TestCase):
    def test_complete_wide_court_is_accepted(self) -> None:
        self.assertTrue(is_complete_court_grid(create_wide_court_grid(), (720, 1280)))

    def test_grid_outside_frame_is_rejected(self) -> None:
        grid = create_wide_court_grid()
        grid[:, :, 0] -= 500

        self.assertFalse(is_complete_court_grid(grid, (720, 1280)))

    def test_top_down_view_without_perspective_is_rejected(self) -> None:
        grid = create_wide_court_grid()
        grid[:, :, 0] = np.linspace(340, 940, 5)

        self.assertFalse(is_complete_court_grid(grid, (720, 1280)))

    def test_small_camera_drift_is_accepted(self) -> None:
        first = create_wide_court_grid()
        second = first + np.array([5.0, 3.0], dtype=np.float32)

        self.assertTrue(is_stable_court_grid_sequence([first, second], (720, 1280)))

    def test_large_camera_drift_is_rejected(self) -> None:
        first = create_wide_court_grid()
        second = first + np.array([100.0, 0.0], dtype=np.float32)

        self.assertFalse(is_stable_court_grid_sequence([first, second], (720, 1280)))


if __name__ == "__main__":
    unittest.main()
