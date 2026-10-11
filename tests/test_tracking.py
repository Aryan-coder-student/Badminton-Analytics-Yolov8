import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ultralytics.engine.results import Boxes

from badminton_analysis.model.detection import Detection
from badminton_analysis.model.segmentation import Segmentation
from badminton_analysis.model.trackers import TRACKING_STRATEGIES, Tracker
from badminton_analysis.model.tracking import TrackedDetection, TrackingModel
from badminton_analysis.model.tracking.inference import ByteTrack
from scratchpad.court_players import (
    create_court_polygon,
    create_player_area_polygon,
    find_segmented_players_inside_court,
)
from scratchpad.test_player_tracking import (
    draw_tracked_player,
    find_available_label_box,
)
from scratchpad.tracked_shot import pair_tracks_with_segmentations


class FakeByteTrackBackend:
    def __init__(self, track_id: int = 7) -> None:
        self.track_id = track_id
        self.inputs: list[Boxes] = []

    def update(self, results: Boxes) -> np.ndarray:
        self.inputs.append(results)
        if not len(results):
            return np.empty((0, 8), dtype=np.float32)
        return np.array(
            [
                [*box, self.track_id, confidence, class_id, index]
                for index, (box, confidence, class_id) in enumerate(
                    zip(results.xyxy, results.conf, results.cls, strict=True),
                )
            ],
            dtype=np.float32,
        )


class FakeTrackingModel(TrackingModel):
    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> list[TrackedDetection]:
        del frame_shape
        return [TrackedDetection(detection, 21) for detection in detections]


class TrackingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.first = Detection((10.0, 20.0, 30.0, 60.0), 0.9, 0, "person")
        self.second = Detection((70.0, 30.0, 90.0, 80.0), 0.8, 0, "person")

    def test_bytetrack_maps_ids_to_original_detections(self) -> None:
        backend = FakeByteTrackBackend(track_id=12)
        tracker = ByteTrack(backend=backend)

        tracks = tracker.update([self.second, self.first], (100, 120))

        self.assertIs(tracks[0].detection, self.second)
        self.assertIs(tracks[1].detection, self.first)
        self.assertEqual([track.track_id for track in tracks], [12, 12])
        np.testing.assert_allclose(
            backend.inputs[0].data,
            [
                [70.0, 30.0, 90.0, 80.0, 0.8, 0.0],
                [10.0, 20.0, 30.0, 60.0, 0.9, 0.0],
            ],
        )
        self.assertEqual(backend.inputs[0].orig_shape, (100, 120))

    def test_bytetrack_preserves_state_across_empty_and_nonempty_frames(self) -> None:
        backend = FakeByteTrackBackend(track_id=4)
        tracker = ByteTrack(backend=backend)

        first_tracks = tracker.update([self.first], (100, 120))
        empty_tracks = tracker.update([], (100, 120))
        next_tracks = tracker.update([self.first], (100, 120))

        self.assertEqual(first_tracks[0].track_id, next_tracks[0].track_id)
        self.assertEqual(empty_tracks, [])
        self.assertEqual(len(backend.inputs), 3)

    def test_tracker_facade_delegates_by_strategy_tag(self) -> None:
        with patch.dict(TRACKING_STRATEGIES, {"fake": FakeTrackingModel}):
            tracker = Tracker("FAKE")

        tracks = tracker.update([self.first], (100, 120))

        self.assertEqual(tracks, [TrackedDetection(self.first, 21)])

    def test_court_filter_runs_before_tracking_and_preserves_mask(self) -> None:
        polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        inside_mask = np.ones((120, 120), dtype=bool)
        inside = Segmentation(self.first, inside_mask)
        outside = Segmentation(
            Detection((110.0, 20.0, 130.0, 60.0), 0.95, 0, "person"),
            np.zeros((120, 120), dtype=bool),
        )
        ball = Segmentation(
            Detection((40.0, 20.0, 50.0, 30.0), 0.99, 32, "sports ball"),
            np.zeros((120, 120), dtype=bool),
        )

        players = find_segmented_players_inside_court(
            [outside, ball, inside],
            polygon,
        )
        tracks = ByteTrack(FakeByteTrackBackend()).update(
            [player.detection for player in players],
            (120, 120),
        )
        paired = pair_tracks_with_segmentations(tracks, players)

        self.assertEqual(len(players), 1)
        self.assertIs(paired[0].segmentation.mask, inside_mask)

    def test_player_area_extends_beyond_court_lines(self) -> None:
        grid = np.zeros((6, 5, 2), dtype=np.float32)
        grid[0, 0] = (20, 20)
        grid[0, -1] = (80, 20)
        grid[-1, -1] = (80, 80)
        grid[-1, 0] = (20, 80)

        court = create_court_polygon(grid)
        player_area = create_player_area_polygon(grid)

        self.assertLess(player_area[0, 0], court[0, 0])
        self.assertGreater(player_area[-1, 1], court[-1, 1])

    @patch("scratchpad.test_player_tracking.cv2.putText")
    def test_visualization_uses_track_id(self, put_text: MagicMock) -> None:
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        segmentation = Segmentation(
            self.first,
            np.zeros((100, 120), dtype=bool),
        )

        tracked_player = pair_tracks_with_segmentations(
            [TrackedDetection(self.first, 42)],
            [segmentation],
        )[0]

        draw_tracked_player(image, tracked_player)

        self.assertEqual(put_text.call_args.args[1], "Player 42")

    def test_label_layout_avoids_an_existing_label(self) -> None:
        first, _ = find_available_label_box(
            (720, 1280),
            (500.0, 200.0, 600.0, 500.0),
            "P1 (track 1)",
            (),
        )

        second, _ = find_available_label_box(
            (720, 1280),
            (520.0, 200.0, 620.0, 500.0),
            "P2 (track 2)",
            (first,),
        )

        self.assertNotEqual(first, second)
        self.assertTrue(first[3] <= second[1] or second[3] <= first[1])


if __name__ == "__main__":
    unittest.main()
