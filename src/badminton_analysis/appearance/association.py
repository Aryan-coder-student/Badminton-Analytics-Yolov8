"""Score possible links between separate local tracking IDs."""

from badminton_analysis.appearance.models import (
    AssociationCandidate,
    AssociationPolicy,
    TrackAssociationProfile,
)
from badminton_analysis.appearance.utils import (
    calculate_court_distance,
    calculate_frame_gap,
    cosine_similarity,
    has_temporal_overlap,
)


def evaluate_association(
    predecessor: TrackAssociationProfile,
    successor: TrackAssociationProfile,
    policy: AssociationPolicy = AssociationPolicy(),
) -> AssociationCandidate:
    """Score one ordered pair without modifying either local track identity."""
    if predecessor.track_id == successor.track_id:
        raise ValueError("A track cannot be associated with itself.")

    frame_gap = calculate_frame_gap(
        predecessor.last_frame,
        successor.first_frame,
    )
    dinov2_similarity = cosine_similarity(
        predecessor.dinov2_descriptor,
        successor.dinov2_descriptor,
    )
    hsv_similarity = cosine_similarity(
        predecessor.hsv_descriptor,
        successor.hsv_descriptor,
    )
    court_distance_meters = calculate_court_distance(
        predecessor.last_court_position,
        successor.first_court_position,
    )
    temporal_overlap = has_temporal_overlap(
        predecessor.first_frame,
        predecessor.last_frame,
        successor.first_frame,
        successor.last_frame,
    )
    temporal_valid = predecessor.last_frame < successor.first_frame
    likely_association = (
        temporal_valid
        and dinov2_similarity >= policy.minimum_dinov2_similarity
        and hsv_similarity >= policy.minimum_hsv_similarity
        and frame_gap <= policy.maximum_frame_gap
        and court_distance_meters <= policy.maximum_court_distance_meters
    )
    return AssociationCandidate(
        predecessor.track_id,
        successor.track_id,
        dinov2_similarity,
        hsv_similarity,
        frame_gap,
        court_distance_meters,
        temporal_overlap,
        temporal_valid,
        likely_association,
    )


def evaluate_track_pairs(
    profiles: list[TrackAssociationProfile],
    policy: AssociationPolicy = AssociationPolicy(),
) -> list[AssociationCandidate]:
    """Evaluate every ordered pair of distinct, unchanged local tracks."""
    return [
        evaluate_association(predecessor, successor, policy)
        for predecessor in profiles
        for successor in profiles
        if predecessor.track_id != successor.track_id
    ]


def rank_valid_predecessors(
    candidates: list[AssociationCandidate],
    successor_track_id: int,
) -> list[AssociationCandidate]:
    """Rank only time-valid predecessors for one later local track."""
    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate.successor_track_id == successor_track_id
        and candidate.temporal_valid
    ]
    return sorted(
        eligible_candidates,
        key=lambda candidate: (
            not candidate.likely_association,
            -candidate.mean_visual_similarity,
            candidate.court_distance_meters,
            candidate.frame_gap,
        ),
    )
