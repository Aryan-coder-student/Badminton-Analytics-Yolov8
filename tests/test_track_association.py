import unittest

import numpy as np

from badminton_analysis.appearance.association import (
    evaluate_association,
    evaluate_track_pairs,
    rank_valid_predecessors,
)
from badminton_analysis.appearance.models import (
    AssociationPolicy,
    TrackAssociationProfile,
)
from badminton_analysis.appearance.utils import (
    calculate_court_distance,
    calculate_frame_gap,
    cosine_similarity,
)
from badminton_analysis.model.detection import Detection
from badminton_analysis.model.segmentation import Segmentation
from badminton_analysis.model.tracking import TrackedSegmentation
from scratchpad.test_track_association import update_track_endpoints
from scratchpad.tracked_shot import TrackedShotFrame


def create_profile(
    track_id: int,
    first_frame: int,
    last_frame: int,
    first_position: tuple[float, float] = (0.0, 0.0),
    last_position: tuple[float, float] = (0.0, 0.0),
    dinov2_descriptor: tuple[float, float] = (1.0, 0.0),
    hsv_descriptor: tuple[float, float] = (0.0, 1.0),
) -> TrackAssociationProfile:
    return TrackAssociationProfile(
        track_id,
        first_frame,
        last_frame,
        np.asarray(first_position, dtype=np.float32),
        np.asarray(last_position, dtype=np.float32),
        np.asarray(dinov2_descriptor, dtype=np.float32),
        np.asarray(hsv_descriptor, dtype=np.float32),
    )


class TrackAssociationTest(unittest.TestCase):
    def test_endpoint_collection_preserves_first_and_latest_positions(self) -> None:
        endpoints = {}
        for frame_number, bbox in (
            (10, (5.0, 5.0, 15.0, 20.0)),
            (12, (7.0, 5.0, 17.0, 22.0)),
        ):
            tracked_player = TrackedSegmentation(
                2,
                Segmentation(
                    Detection(bbox, 1.0, 0, "person"),
                    np.ones((32, 32), dtype=bool),
                ),
            )
            tracked_frame = TrackedShotFrame(
                frame_number,
                30.0,
                np.zeros((32, 32, 3), dtype=np.uint8),
                np.zeros((6, 5, 2), dtype=np.float32),
                [tracked_player],
            )
            update_track_endpoints(endpoints, tracked_frame, np.eye(3))

        self.assertEqual((endpoints[2].first_frame, endpoints[2].last_frame), (10, 12))
        np.testing.assert_allclose(endpoints[2].first_court_position, [10.0, 20.0])
        np.testing.assert_allclose(endpoints[2].last_court_position, [12.0, 22.0])

    def test_overlapping_tracks_are_rejected(self) -> None:
        predecessor = create_profile(1, 10, 20)
        successor = create_profile(2, 15, 25)

        candidate = evaluate_association(predecessor, successor)

        self.assertTrue(candidate.temporal_overlap)
        self.assertFalse(candidate.temporal_valid)
        self.assertFalse(candidate.likely_association)

    def test_frame_gap_is_elapsed_from_last_to_first_frame(self) -> None:
        self.assertEqual(calculate_frame_gap(100, 101), 1)
        self.assertEqual(calculate_frame_gap(100, 105), 5)

    def test_cosine_similarity_is_calculated_for_descriptor_vectors(self) -> None:
        similarity = cosine_similarity(
            np.array([1.0, 0.0]),
            np.array([1.0, 1.0]),
        )

        self.assertAlmostEqual(similarity, 1 / np.sqrt(2), places=6)

    def test_court_distance_uses_euclidean_metres(self) -> None:
        distance = calculate_court_distance(
            np.array([0.0, 0.0]),
            np.array([3.0, 4.0]),
        )

        self.assertEqual(distance, 5.0)

    def test_tracks_two_and_five_are_evaluated_without_identity_merging(self) -> None:
        track_two = create_profile(
            2,
            10,
            20,
            last_position=(2.0, 3.0),
        )
        track_five = create_profile(
            5,
            23,
            35,
            first_position=(2.5, 3.5),
        )

        candidates = evaluate_track_pairs([track_two, track_five])
        forward_candidate = next(
            candidate
            for candidate in candidates
            if candidate.predecessor_track_id == 2
            and candidate.successor_track_id == 5
        )

        self.assertEqual({track_two.track_id, track_five.track_id}, {2, 5})
        self.assertEqual(len(candidates), 2)
        self.assertTrue(forward_candidate.temporal_valid)
        self.assertTrue(forward_candidate.likely_association)
        self.assertEqual(
            [candidate.predecessor_track_id for candidate in rank_valid_predecessors(
                candidates,
                successor_track_id=5,
            )],
            [2],
        )

    def test_policy_can_report_visual_match_as_not_likely_when_too_far(self) -> None:
        predecessor = create_profile(2, 10, 20, last_position=(0.0, 0.0))
        successor = create_profile(5, 25, 30, first_position=(5.0, 0.0))

        candidate = evaluate_association(
            predecessor,
            successor,
            AssociationPolicy(maximum_court_distance_meters=4.0),
        )

        self.assertTrue(candidate.temporal_valid)
        self.assertFalse(candidate.likely_association)


if __name__ == "__main__":
    unittest.main()
