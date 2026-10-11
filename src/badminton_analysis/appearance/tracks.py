from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from badminton_analysis.appearance.features import (
    extract_appearance_input,
    normalize_descriptor,
)
from badminton_analysis.appearance.models import AppearanceInput
from badminton_analysis.model.embeddings import Embedding
from badminton_analysis.model.tracking import TrackedSegmentation


class BatchEmbedder(Protocol):
    """Embedding operation required by track appearance extraction."""

    def encode_batch(self, inputs: list[object]) -> np.ndarray:
        """Encode one sampled frame's masked player crops."""


@dataclass(frozen=True, slots=True)
class TrackAppearanceConfig:
    """Sampling policy for track appearance extraction."""

    sampling_interval: int = 5

    def __post_init__(self) -> None:
        if self.sampling_interval < 1:
            raise ValueError("sampling_interval must be at least 1.")


@dataclass(frozen=True, slots=True, eq=False)
class AppearanceObservation:
    """Appearance features tied to one exact tracked segmentation and frame."""

    frame_number: int
    tracked_player: TrackedSegmentation
    masked_crop: np.ndarray
    dinov2_embedding: np.ndarray
    hsv_descriptor: np.ndarray

    @property
    def track_id(self) -> int:
        return self.tracked_player.track_id


@dataclass(slots=True)
class TrackAppearance:
    """Sampled observations and final descriptors for one local track ID."""

    track_id: int
    observations: list[AppearanceObservation] = field(default_factory=list)
    final_dinov2_descriptor: np.ndarray | None = None
    final_hsv_descriptor: np.ndarray | None = None

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame_number

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame_number

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def dinov2_embeddings(self) -> list[np.ndarray]:
        return [observation.dinov2_embedding for observation in self.observations]

    @property
    def hsv_descriptors(self) -> list[np.ndarray]:
        return [observation.hsv_descriptor for observation in self.observations]

    def add_observation(self, observation: AppearanceObservation) -> None:
        if observation.track_id != self.track_id:
            raise ValueError("An observation cannot be added to another track ID.")
        self.observations.append(observation)
        self.final_dinov2_descriptor = None
        self.final_hsv_descriptor = None

    def aggregate(self) -> None:
        """Aggregate all observations into one normalized descriptor per type."""
        self.final_dinov2_descriptor = normalize_descriptor(
            np.mean(self.dinov2_embeddings, axis=0),
        )
        self.final_hsv_descriptor = normalize_descriptor(
            np.mean(self.hsv_descriptors, axis=0),
        )


@dataclass(frozen=True, slots=True)
class _PreparedObservation:
    frame_number: int
    tracked_player: TrackedSegmentation
    appearance_input: AppearanceInput


@dataclass
class TrackAppearanceCollector:
    """Sample and aggregate appearance descriptors by local ByteTrack ID."""

    embedder: BatchEmbedder = field(
        default_factory=lambda: Embedding("dinov2"),
        repr=False,
    )
    config: TrackAppearanceConfig = TrackAppearanceConfig()
    _tracks: dict[int, TrackAppearance] = field(default_factory=dict, init=False)
    _last_sample_frames: dict[int, int] = field(default_factory=dict, init=False)

    def add_frame(
        self,
        frame_number: int,
        image: np.ndarray,
        tracked_players: list[TrackedSegmentation],
    ) -> list[AppearanceObservation]:
        prepared = self._prepare_observations(
            frame_number,
            image,
            tracked_players,
        )
        if not prepared:
            return []

        embeddings = self.embedder.encode_batch(
            [item.appearance_input.masked_crop for item in prepared],
        )
        observations = [
            AppearanceObservation(
                frame_number=item.frame_number,
                tracked_player=item.tracked_player,
                masked_crop=item.appearance_input.masked_crop,
                dinov2_embedding=normalize_descriptor(embedding),
                hsv_descriptor=item.appearance_input.hsv_descriptor,
            )
            for item, embedding in zip(prepared, embeddings, strict=True)
        ]
        for observation in observations:
            self._store_observation(observation)
        return observations

    def _prepare_observations(
        self,
        frame_number: int,
        image: np.ndarray,
        tracked_players: list[TrackedSegmentation],
    ) -> list[_PreparedObservation]:
        prepared: list[_PreparedObservation] = []
        for tracked_player in tracked_players:
            if not self._should_sample(tracked_player.track_id, frame_number):
                continue
            appearance_input = extract_appearance_input(
                image,
                tracked_player.segmentation,
            )
            if appearance_input is not None:
                prepared.append(
                    _PreparedObservation(
                        frame_number,
                        tracked_player,
                        appearance_input,
                    )
                )
        return prepared

    def _should_sample(self, track_id: int, frame_number: int) -> bool:
        last_frame = self._last_sample_frames.get(track_id)
        return (
            last_frame is None
            or frame_number - last_frame >= self.config.sampling_interval
        )

    def _store_observation(self, observation: AppearanceObservation) -> None:
        track = self._tracks.get(observation.track_id)
        if track is None:
            track = TrackAppearance(observation.track_id)
            self._tracks[observation.track_id] = track
        track.add_observation(observation)
        self._last_sample_frames[observation.track_id] = observation.frame_number

    def finalize(self) -> dict[int, TrackAppearance]:
        """Aggregate and return local tracks without performing identity matching."""
        for track in self._tracks.values():
            track.aggregate()
        return dict(self._tracks)
