"""Reproduce visual shot clustering from existing camera-cut boundaries.

Inputs are the shot *intervals* produced by a separate cut detector (TransNetV2).
The cluster is an unsupervised visual group, NOT a semantic camera-view label.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from sklearn.metrics import silhouette_score


@dataclass(frozen=True, slots=True)
class ShotRange:
    """Zero-based [start_frame, end_frame) from the shot-boundary detector."""

    shot_id: str
    start_frame: int
    end_frame: int


class ImageEmbedder(Protocol):
    def encode_batch(self, inputs: list[np.ndarray]) -> np.ndarray: ...


def load_shot_ranges(path: Path) -> list[ShotRange]:
    """Read validated, nonoverlapping shot intervals from a CSV export."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"shot_id", "start_frame", "end_frame"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Expected shot-range columns: {sorted(required)}")
        result = [
            ShotRange(row["shot_id"].strip(), int(row["start_frame"]), int(row["end_frame"]))
            for row in reader
        ]
    if not result:
        raise ValueError("Shot-range CSV is empty")
    if len({shot.shot_id for shot in result}) != len(result):
        raise ValueError("Duplicate shot_id in shot-range CSV")
    if any(
        not shot.shot_id or shot.start_frame < 0 or shot.end_frame <= shot.start_frame
        for shot in result
    ):
        raise ValueError("Shot IDs must be nonempty and frame ranges positive")
    result.sort(key=lambda shot: shot.start_frame)
    if any(
        previous.end_frame > next_shot.start_frame
        for previous, next_shot in zip(result, result[1:])
    ):
        raise ValueError("Shot ranges must not overlap")
    return result


def interior_sample_indices(shot: ShotRange) -> tuple[int, ...]:
    """Sample 25%, 50%, 75% of a shot, keeping short shots unique."""
    last_offset = shot.end_frame - shot.start_frame - 1
    indices = [
        shot.start_frame + round(last_offset * fraction)
        for fraction in (0.25, 0.5, 0.75)
    ]
    return tuple(dict.fromkeys(indices))


def extract_shot_embeddings(
    video_path: Path,
    shot_ranges: list[ShotRange],
    embedder: ImageEmbedder,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Mean the sampled-frame embeddings for each camera shot.

    OpenCV decodes BGR images, while the project's embedding strategy takes RGB.
    Reuse one embedding model across all shots; do not run tracking or ReID.
    """
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Cannot open source video: {video_path}")
    summaries: list[np.ndarray] = []
    index_groups: list[tuple[int, ...]] = []
    try:
        for shot in shot_ranges:
            indices = interior_sample_indices(shot)
            frames: list[np.ndarray] = []
            for index in indices:
                if not video.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise ValueError(f"Failed to seek video to frame {index}")
                ok, frame = video.read()
                if not ok:
                    raise ValueError(
                        f"Cannot read video frame {index} (shot {shot.shot_id})"
                    )
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            embeddings = np.asarray(
                embedder.encode_batch(frames), dtype=np.float32
            )
            if (
                embeddings.ndim != 2
                or embeddings.shape[0] != len(frames)
                or not np.isfinite(embeddings).all()
            ):
                raise ValueError(f"Invalid embedding batch for shot {shot.shot_id}")
            summaries.append(np.mean(embeddings, axis=0))
            index_groups.append(indices)
    finally:
        video.release()
    return np.stack(summaries), index_groups


def save_embedding_cache(
    path: Path,
    shots: list[ShotRange],
    embeddings: np.ndarray,
    sample_indices: list[tuple[int, ...]],
) -> None:
    """Store shot-level features without pickled Python objects."""
    if len(shots) != len(embeddings) or len(shots) != len(sample_indices):
        raise ValueError("Shot, embedding, and sampled-frame counts must agree")
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.full((len(shots), 3), -1, dtype=np.int64)
    for row, indices in enumerate(sample_indices):
        samples[row, :len(indices)] = indices
    np.savez_compressed(
        path,
        shot_ids=np.asarray([shot.shot_id for shot in shots], dtype=str),
        start_frames=np.asarray(
            [shot.start_frame for shot in shots], dtype=np.int64
        ),
        end_frames=np.asarray([shot.end_frame for shot in shots], dtype=np.int64),
        sample_frames=samples,
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )


def load_embedding_cache(
    path: Path,
) -> tuple[list[ShotRange], np.ndarray, list[tuple[int, ...]]]:
    """Load the checked shot range and sample provenance from cache."""
    with np.load(path, allow_pickle=False) as data:
        shots = [
            ShotRange(str(shot_id), int(start), int(end))
            for shot_id, start, end in zip(
                data["shot_ids"], data["start_frames"], data["end_frames"],
                strict=True,
            )
        ]
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        indices = [
            tuple(int(value) for value in row if value >= 0)
            for row in data["sample_frames"]
        ]
    if (
        embeddings.ndim != 2
        or len(shots) != len(embeddings)
        or len(shots) != len(indices)
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("Invalid cached embedding matrix")
    if len({shot.shot_id for shot in shots}) != len(shots):
        raise ValueError("Duplicate shot IDs in embedding cache")
    if any(
        shot.start_frame < 0
        or shot.end_frame <= shot.start_frame
        or not indices[i]
        for i, shot in enumerate(shots)
    ):
        raise ValueError("Invalid shot ranges or samples in embedding cache")
    return shots, embeddings, indices


def group_shots(
    embeddings: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, float | None]:
    """Run the existing swappable KMeans strategy, with reproducible seed."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("Expected a finite 2D shot embedding matrix")
    if not 2 <= n_clusters <= len(embeddings):
        raise ValueError("n_clusters must be between 2 and the number of shots")
    from badminton_analysis.model.clusters import Cluster

    assignments = Cluster(
        "kmeans",
        embeddings,
        options={"n_clusters": n_clusters, "random_state": 42},
    ).fit_predict()
    labels = np.asarray(assignments, dtype=np.int64)
    score = (
        float(silhouette_score(embeddings, labels))
        if 1 < len(set(labels.tolist())) < len(embeddings)
        else None
    )
    return labels, score


def write_shot_clusters(
    path: Path,
    shots: list[ShotRange],
    labels: np.ndarray,
    sample_indices: list[tuple[int, ...]],
) -> None:
    """Export the CSV expected by camera_views.load_shot_clusters."""
    if len(shots) != len(labels) or len(shots) != len(sample_indices):
        raise ValueError("Cluster export counts do not agree")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "shot_id", "cluster_id", "start_frame", "end_frame",
            "sample25_frame", "sample50_frame", "sample75_frame",
        ])
        writer.writeheader()
        for shot, label, samples in zip(
            shots, labels, sample_indices, strict=True
        ):
            cells = list(samples) + [""] * (3 - len(samples))
            writer.writerow({
                "shot_id": shot.shot_id,
                "cluster_id": int(label),
                "start_frame": shot.start_frame,
                "end_frame": shot.end_frame,
                "sample25_frame": cells[0],
                "sample50_frame": cells[1],
                "sample75_frame": cells[2],
            })
