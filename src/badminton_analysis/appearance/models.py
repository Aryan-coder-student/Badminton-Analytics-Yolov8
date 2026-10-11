from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class AppearanceInput:
    """Validated DINOv2 image input and mask-aware color descriptor."""

    masked_crop: np.ndarray
    hsv_descriptor: np.ndarray


@dataclass(frozen=True, slots=True)
class AssociationPolicy:
    """Conservative limits for reporting a likely local-track continuation."""

    minimum_dinov2_similarity: float = 0.85
    minimum_hsv_similarity: float = 0.90
    maximum_frame_gap: int = 60
    maximum_court_distance_meters: float = 4.0


@dataclass(frozen=True, slots=True, eq=False)
class TrackAssociationProfile:
    """Final appearance and endpoint evidence for one unchanged local track."""

    track_id: int
    first_frame: int
    last_frame: int
    first_court_position: np.ndarray
    last_court_position: np.ndarray
    dinov2_descriptor: np.ndarray
    hsv_descriptor: np.ndarray


@dataclass(frozen=True, slots=True)
class AssociationCandidate:
    """Diagnostic evidence for one ordered predecessor-successor pair."""

    predecessor_track_id: int
    successor_track_id: int
    dinov2_similarity: float
    hsv_similarity: float
    frame_gap: int
    court_distance_meters: float
    temporal_overlap: bool
    temporal_valid: bool
    likely_association: bool

    @property
    def mean_visual_similarity(self) -> float:
        """Return the equally weighted DINOv2 and HSV score used for ranking."""
        return (self.dinov2_similarity + self.hsv_similarity) / 2


@dataclass(frozen=True, slots=True)
class PlayerIdentity:
    """One anonymous player represented by one or more local ByteTrack IDs."""

    player_id: str
    local_track_ids: frozenset[int]
    first_frame: int
    last_frame: int


@dataclass(frozen=True, slots=True, eq=False)
class TrackAppearanceGallery:
    """Multiple normalized appearance observations for one shot-local track."""

    shot_id: int
    track_id: int
    frame_numbers: np.ndarray
    dinov2_embeddings: np.ndarray
    hsv_descriptors: np.ndarray

    @property
    def observation_count(self) -> int:
        return len(self.frame_numbers)


@dataclass(frozen=True, slots=True, eq=False)
class PersistentPlayerGallery:
    """Bounded match-level appearance gallery for one anonymous player."""

    player_id: str
    source_track_keys: frozenset[tuple[int, int]]
    frame_numbers: np.ndarray
    dinov2_embeddings: np.ndarray
    hsv_descriptors: np.ndarray

    @property
    def observation_count(self) -> int:
        return len(self.frame_numbers)


@dataclass(frozen=True, slots=True)
class CrossShotReIDConfig:
    """Controls robust appearance scoring without a fixed match threshold."""

    top_k: int = 5
    dino_weight: float = 0.8
    hsv_weight: float = 0.2

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if self.dino_weight < 0 or self.hsv_weight < 0:
            raise ValueError("Appearance weights cannot be negative.")
        if not np.isclose(self.dino_weight + self.hsv_weight, 1.0):
            raise ValueError("DINOv2 and HSV weights must sum to 1.")


@dataclass(frozen=True, slots=True)
class GalleryScore:
    """Robust DINOv2, HSV, and weighted appearance scores for one gallery pair."""

    dinov2_score: float
    hsv_score: float
    appearance_score: float


@dataclass(frozen=True, slots=True, eq=False)
class CrossShotSimilarity:
    """All gallery scores between target tracks and persistent players."""

    target_track_ids: tuple[int, ...]
    source_player_ids: tuple[str, ...]
    dinov2_matrix: np.ndarray
    hsv_matrix: np.ndarray
    appearance_matrix: np.ndarray


@dataclass(frozen=True, slots=True)
class CrossShotAssignment:
    """One proposed target-track to persistent-player match with confidence data."""

    target_track_id: int
    player_id: str
    dinov2_score: float
    hsv_score: float
    appearance_score: float
    second_best_player_id: str | None
    second_best_score: float | None
    margin: float | None
