import unittest

import numpy as np

from badminton_analysis.appearance.gallery import (
    assign_cross_shot_players,
    build_cross_shot_similarity,
    create_persistent_galleries,
    pairwise_cosine_similarity,
    score_gallery_pair,
    similarity_to_cost,
    top_k_mean,
    update_persistent_galleries,
)
from badminton_analysis.appearance.models import (
    CrossShotAssignment,
    CrossShotReIDConfig,
    CrossShotSimilarity,
    PlayerIdentity,
    PersistentPlayerGallery,
    TrackAppearanceGallery,
)


def create_track_gallery(
    shot_id: int,
    track_id: int,
    embeddings: tuple[tuple[float, float], ...],
) -> TrackAppearanceGallery:
    values = np.asarray(embeddings, dtype=np.float32)
    return TrackAppearanceGallery(
        shot_id,
        track_id,
        np.arange(len(values)),
        values,
        values,
    )


def create_player_gallery(
    player_id: str,
    embeddings: tuple[tuple[float, float], ...],
) -> PersistentPlayerGallery:
    values = np.asarray(embeddings, dtype=np.float32)
    return PersistentPlayerGallery(
        player_id,
        frozenset(),
        np.arange(len(values)),
        values,
        values,
    )


class CrossShotReIDTest(unittest.TestCase):
    def test_pairwise_gallery_cosine_matrix(self) -> None:
        matrix = pairwise_cosine_similarity(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[1.0, 0.0], [1.0, 1.0]]),
        )

        np.testing.assert_allclose(matrix, [[1.0, 1.0], [0.0, 1.0]])

    def test_top_k_uses_only_strongest_comparisons(self) -> None:
        score = top_k_mean(np.array([[0.2, 0.9], [0.8, 0.7]]), top_k=2)

        self.assertAlmostEqual(score, 0.85)

    def test_top_k_uses_all_comparisons_when_gallery_is_smaller(self) -> None:
        score = top_k_mean(np.array([[0.2, 0.8]]), top_k=5)

        self.assertAlmostEqual(score, 0.5)

    def test_weighted_dino_and_hsv_scores_are_combined(self) -> None:
        first = create_track_gallery(18, 1, ((1.0, 0.0),))
        second = create_player_gallery("P1", ((0.5, 0.5),))
        first = TrackAppearanceGallery(
            first.shot_id,
            first.track_id,
            first.frame_numbers,
            first.dinov2_embeddings,
            np.array([[0.0, 1.0]], dtype=np.float32),
        )
        second = PersistentPlayerGallery(
            second.player_id,
            second.source_track_keys,
            second.frame_numbers,
            second.dinov2_embeddings,
            np.array([[0.0, 1.0]], dtype=np.float32),
        )

        score = score_gallery_pair(first, second)

        self.assertAlmostEqual(score.dinov2_score, 0.5)
        self.assertAlmostEqual(score.hsv_score, 1.0)
        self.assertAlmostEqual(score.appearance_score, 0.6)

    def test_invalid_weight_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CrossShotReIDConfig(dino_weight=0.7, hsv_weight=0.2)

    def test_similarity_is_converted_to_hungarian_cost(self) -> None:
        cost = similarity_to_cost(np.array([[0.9, 0.2]]))

        np.testing.assert_allclose(cost, [[0.1, 0.8]], atol=1e-6)

    def test_hungarian_assignment_is_one_to_one(self) -> None:
        similarity = CrossShotSimilarity(
            (1, 2),
            ("P1", "P2"),
            np.array([[0.1, 0.9], [0.8, 0.2]]),
            np.array([[0.1, 0.9], [0.8, 0.2]]),
            np.array([[0.1, 0.9], [0.8, 0.2]]),
        )

        assignments = assign_cross_shot_players(similarity)

        self.assertEqual(
            [(item.target_track_id, item.player_id) for item in assignments],
            [(1, "P2"), (2, "P1")],
        )
        self.assertEqual({item.player_id for item in assignments}, {"P1", "P2"})

    def test_assignment_reports_second_best_score_and_margin(self) -> None:
        similarity = CrossShotSimilarity(
            (1,),
            ("P1", "P2"),
            np.array([[0.9, 0.6]]),
            np.array([[0.9, 0.6]]),
            np.array([[0.9, 0.6]]),
        )

        assignment = assign_cross_shot_players(similarity)[0]

        self.assertEqual(assignment.player_id, "P1")
        self.assertEqual(assignment.second_best_player_id, "P2")
        self.assertAlmostEqual(assignment.second_best_score or 0, 0.6)
        self.assertAlmostEqual(assignment.margin or 0, 0.3)

    def test_raw_track_ids_remain_unchanged_after_gallery_update(self) -> None:
        source_track = create_track_gallery(18, 2, ((1.0, 0.0),))
        target_track = create_track_gallery(23, 5, ((1.0, 0.0),))
        identity = PlayerIdentity("P1", frozenset({2}), 0, 10)
        persistent = create_persistent_galleries(
            [source_track],
            {2: identity},
            maximum_observations=5,
        )
        assignment = assign_cross_shot_players(
            build_cross_shot_similarity([target_track], persistent),
        )[0]

        updated, accepted = update_persistent_galleries(
            persistent,
            [target_track],
            [assignment],
            maximum_observations=5,
        )

        self.assertEqual((source_track.track_id, target_track.track_id), (2, 5))
        self.assertEqual(accepted, {5: "P1"})
        self.assertEqual(updated[0].source_track_keys, frozenset({(18, 2), (23, 5)}))

    def test_low_score_assignment_does_not_update_a_player_gallery(self) -> None:
        source_track = create_track_gallery(18, 2, ((1.0, 0.0),))
        target_track = create_track_gallery(23, 5, ((0.5, 0.5),))
        persistent = create_persistent_galleries(
            [source_track],
            {2: PlayerIdentity("P1", frozenset({2}), 0, 10)},
            maximum_observations=5,
        )
        low_score_assignment = CrossShotAssignment(
            5,
            "P1",
            0.6,
            0.6,
            0.6,
            None,
            None,
            None,
        )

        updated, accepted = update_persistent_galleries(
            persistent,
            [target_track],
            [low_score_assignment],
            maximum_observations=5,
            minimum_appearance_score=0.7,
        )

        self.assertEqual(accepted, {})
        self.assertEqual(updated[0].source_track_keys, frozenset({(18, 2)}))

if __name__ == "__main__":
    unittest.main()
