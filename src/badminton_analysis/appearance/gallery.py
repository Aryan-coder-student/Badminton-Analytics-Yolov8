"""Pure gallery comparison and one-to-one cross-shot ReID assignment."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from badminton_analysis.appearance.models import (
    CrossShotAssignment,
    CrossShotReIDConfig,
    CrossShotSimilarity,
    GalleryScore,
    PersistentPlayerGallery,
    PlayerIdentity,
    TrackAppearanceGallery,
)


def pairwise_cosine_similarity(
    first_embeddings: np.ndarray,
    second_embeddings: np.ndarray,
) -> np.ndarray:
    """Return every cosine similarity between two normalized galleries."""
    first = np.asarray(first_embeddings, dtype=np.float32)
    second = np.asarray(second_embeddings, dtype=np.float32)
    return first @ second.T


def top_k_mean(similarity_matrix: np.ndarray, top_k: int) -> float:
    """Return the mean of the strongest available gallery comparisons."""
    similarities = np.asarray(similarity_matrix, dtype=np.float32).reshape(-1)
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if similarities.size == 0:
        raise ValueError("A gallery comparison requires at least one pair.")
    count = min(top_k, similarities.size)
    strongest = np.partition(similarities, -count)[-count:]
    return float(np.mean(strongest))


def score_gallery_pair(
    first: TrackAppearanceGallery | PersistentPlayerGallery,
    second: TrackAppearanceGallery | PersistentPlayerGallery,
    config: CrossShotReIDConfig = CrossShotReIDConfig(),
) -> GalleryScore:
    """Score two multi-observation galleries without frame-to-frame matching."""
    dinov2_score = top_k_mean(
        pairwise_cosine_similarity(first.dinov2_embeddings, second.dinov2_embeddings),
        config.top_k,
    )
    hsv_score = top_k_mean(
        pairwise_cosine_similarity(first.hsv_descriptors, second.hsv_descriptors),
        config.top_k,
    )
    appearance_score = (
        config.dino_weight * dinov2_score + config.hsv_weight * hsv_score
    )
    return GalleryScore(dinov2_score, hsv_score, appearance_score)


def build_cross_shot_similarity(
    target_galleries: list[TrackAppearanceGallery],
    source_galleries: list[PersistentPlayerGallery],
    config: CrossShotReIDConfig = CrossShotReIDConfig(),
) -> CrossShotSimilarity:
    """Build DINOv2, HSV, and combined target-track by player matrices."""
    ordered_targets = sorted(target_galleries, key=lambda gallery: gallery.track_id)
    ordered_sources = sorted(source_galleries, key=lambda gallery: gallery.player_id)
    scores = [
        [score_gallery_pair(target, source, config) for source in ordered_sources]
        for target in ordered_targets
    ]
    return CrossShotSimilarity(
        tuple(gallery.track_id for gallery in ordered_targets),
        tuple(gallery.player_id for gallery in ordered_sources),
        np.asarray([[score.dinov2_score for score in row] for row in scores]),
        np.asarray([[score.hsv_score for score in row] for row in scores]),
        np.asarray([[score.appearance_score for score in row] for row in scores]),
    )


def similarity_to_cost(similarity_matrix: np.ndarray) -> np.ndarray:
    """Convert the maximum-similarity objective to Hungarian assignment cost."""
    return 1.0 - np.asarray(similarity_matrix, dtype=np.float32)


def assign_cross_shot_players(
    similarity: CrossShotSimilarity,
) -> list[CrossShotAssignment]:
    """Find one-to-one proposed matches and report each row's next-best option."""
    row_indices, column_indices = linear_sum_assignment(
        similarity_to_cost(similarity.appearance_matrix),
    )
    assignments = [
        _create_assignment(similarity, row_index, column_index)
        for row_index, column_index in zip(row_indices, column_indices, strict=True)
    ]
    return sorted(assignments, key=lambda assignment: assignment.target_track_id)


def _create_assignment(
    similarity: CrossShotSimilarity,
    row_index: int,
    column_index: int,
) -> CrossShotAssignment:
    second_best_index = _second_best_index(
        similarity.appearance_matrix[row_index],
        column_index,
    )
    second_best_score = (
        None
        if second_best_index is None
        else float(similarity.appearance_matrix[row_index, second_best_index])
    )
    margin = (
        None
        if second_best_score is None
        else float(similarity.appearance_matrix[row_index, column_index])
        - second_best_score
    )
    return CrossShotAssignment(
        similarity.target_track_ids[row_index],
        similarity.source_player_ids[column_index],
        float(similarity.dinov2_matrix[row_index, column_index]),
        float(similarity.hsv_matrix[row_index, column_index]),
        float(similarity.appearance_matrix[row_index, column_index]),
        None
        if second_best_index is None
        else similarity.source_player_ids[second_best_index],
        second_best_score,
        margin,
    )


def _second_best_index(scores: np.ndarray, assigned_index: int) -> int | None:
    candidate_indices = [
        index for index in range(len(scores)) if index != assigned_index
    ]
    if not candidate_indices:
        return None
    return max(candidate_indices, key=lambda index: scores[index])


def create_persistent_galleries(
    track_galleries: list[TrackAppearanceGallery],
    identities_by_track_id: dict[int, PlayerIdentity],
    maximum_observations: int,
) -> list[PersistentPlayerGallery]:
    """Create one bounded persistent gallery for every initial logical player."""
    galleries_by_player_id: dict[str, list[TrackAppearanceGallery]] = {}
    for track_gallery in track_galleries:
        identity = identities_by_track_id[track_gallery.track_id]
        galleries_by_player_id.setdefault(identity.player_id, []).append(track_gallery)
    return [
        _combine_track_galleries(player_id, galleries, maximum_observations)
        for player_id, galleries in sorted(galleries_by_player_id.items())
    ]


def update_persistent_galleries(
    persistent_galleries: list[PersistentPlayerGallery],
    target_galleries: list[TrackAppearanceGallery],
    assignments: list[CrossShotAssignment],
    maximum_observations: int,
    minimum_margin: float = 0.0,
    minimum_appearance_score: float = 0.0,
) -> tuple[list[PersistentPlayerGallery], dict[int, str]]:
    """Add only sufficiently strong, non-ambiguous matches to galleries."""
    gallery_by_player_id = {
        gallery.player_id: gallery for gallery in persistent_galleries
    }
    target_by_track_id = {gallery.track_id: gallery for gallery in target_galleries}
    accepted_player_by_track_id: dict[int, str] = {}
    for assignment in assignments:
        if not _is_confident_assignment(
            assignment,
            minimum_margin,
            minimum_appearance_score,
        ):
            continue
        persistent_gallery = gallery_by_player_id[assignment.player_id]
        target_gallery = target_by_track_id[assignment.target_track_id]
        gallery_by_player_id[assignment.player_id] = _append_track_gallery(
            persistent_gallery,
            target_gallery,
            maximum_observations,
        )
        accepted_player_by_track_id[assignment.target_track_id] = assignment.player_id
    return (
        [
            gallery_by_player_id[player_id]
            for player_id in sorted(gallery_by_player_id)
        ],
        accepted_player_by_track_id,
    )


def _is_confident_assignment(
    assignment: CrossShotAssignment,
    minimum_margin: float,
    minimum_appearance_score: float,
) -> bool:
    return (
        assignment.appearance_score >= minimum_appearance_score
        and (
            assignment.margin is None
            or assignment.margin > minimum_margin
        )
    )


def _combine_track_galleries(
    player_id: str,
    track_galleries: list[TrackAppearanceGallery],
    maximum_observations: int,
) -> PersistentPlayerGallery:
    return _create_persistent_gallery(
        player_id,
        frozenset(
            (gallery.shot_id, gallery.track_id) for gallery in track_galleries
        ),
        np.concatenate([gallery.frame_numbers for gallery in track_galleries]),
        np.concatenate(
            [gallery.dinov2_embeddings for gallery in track_galleries],
        ),
        np.concatenate([gallery.hsv_descriptors for gallery in track_galleries]),
        maximum_observations,
    )


def _append_track_gallery(
    persistent_gallery: PersistentPlayerGallery,
    target_gallery: TrackAppearanceGallery,
    maximum_observations: int,
) -> PersistentPlayerGallery:
    return _create_persistent_gallery(
        persistent_gallery.player_id,
        persistent_gallery.source_track_keys
        | {(target_gallery.shot_id, target_gallery.track_id)},
        np.concatenate(
            (persistent_gallery.frame_numbers, target_gallery.frame_numbers),
        ),
        np.concatenate(
            (
                persistent_gallery.dinov2_embeddings,
                target_gallery.dinov2_embeddings,
            ),
        ),
        np.concatenate(
            (persistent_gallery.hsv_descriptors, target_gallery.hsv_descriptors),
        ),
        maximum_observations,
    )


def _create_persistent_gallery(
    player_id: str,
    source_track_keys: frozenset[tuple[int, int]],
    frame_numbers: np.ndarray,
    dinov2_embeddings: np.ndarray,
    hsv_descriptors: np.ndarray,
    maximum_observations: int,
) -> PersistentPlayerGallery:
    if maximum_observations < 1:
        raise ValueError("maximum_observations must be at least 1.")
    indices = _representative_indices(len(frame_numbers), maximum_observations)
    return PersistentPlayerGallery(
        player_id,
        source_track_keys,
        frame_numbers[indices],
        dinov2_embeddings[indices],
        hsv_descriptors[indices],
    )


def _representative_indices(count: int, maximum_observations: int) -> np.ndarray:
    if count <= maximum_observations:
        return np.arange(count)
    return np.linspace(0, count - 1, maximum_observations, dtype=int)
