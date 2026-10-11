"""Track-level player appearance extraction."""

from badminton_analysis.appearance.association import (
    evaluate_association,
    evaluate_track_pairs,
    rank_valid_predecessors,
)
from badminton_analysis.appearance.identity import (
    propagate_local_player_ids,
    resolve_player_identities,
)
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
from badminton_analysis.appearance.features import (
    extract_appearance_input,
    normalize_descriptor,
)
from badminton_analysis.appearance.models import (
    AppearanceInput,
    AssociationCandidate,
    AssociationPolicy,
    CrossShotAssignment,
    CrossShotReIDConfig,
    CrossShotSimilarity,
    GalleryScore,
    PersistentPlayerGallery,
    PlayerIdentity,
    TrackAppearanceGallery,
    TrackAssociationProfile,
)
from badminton_analysis.appearance.tracks import (
    AppearanceObservation,
    TrackAppearance,
    TrackAppearanceCollector,
    TrackAppearanceConfig,
)
from badminton_analysis.appearance.utils import (
    calculate_court_distance,
    calculate_frame_gap,
    cosine_similarity,
)

__all__ = [
    "AssociationCandidate",
    "AssociationPolicy",
    "AppearanceInput",
    "AppearanceObservation",
    "CrossShotAssignment",
    "CrossShotReIDConfig",
    "CrossShotSimilarity",
    "GalleryScore",
    "PersistentPlayerGallery",
    "PlayerIdentity",
    "TrackAppearance",
    "TrackAppearanceCollector",
    "TrackAppearanceConfig",
    "TrackAppearanceGallery",
    "TrackAssociationProfile",
    "calculate_court_distance",
    "calculate_frame_gap",
    "cosine_similarity",
    "evaluate_association",
    "evaluate_track_pairs",
    "extract_appearance_input",
    "normalize_descriptor",
    "pairwise_cosine_similarity",
    "rank_valid_predecessors",
    "propagate_local_player_ids",
    "resolve_player_identities",
    "score_gallery_pair",
    "similarity_to_cost",
    "top_k_mean",
    "assign_cross_shot_players",
    "build_cross_shot_similarity",
    "create_persistent_galleries",
    "update_persistent_galleries",
]
