"""Attach reviewed shot-cluster camera views to player benchmark annotations.

This consumes existing shot clustering outputs. It does NOT run embeddings,
fit clustering models, or assume that cluster numbers imply semantic views.
"""

import csv
import json
from pathlib import Path


CAMERA_VIEWS = frozenset({
    "wide",
    "partial_court",
    "closeup",
    "side_angle",
    "transition",
    "other",
    "needs_review",
})


def load_shot_clusters(path: Path) -> list[dict[str, str | int]]:
    """Read shot cluster assignments with absolute [start_frame, end_frame) bounds."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"shot_id", "cluster_id", "start_frame", "end_frame"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Shot cluster CSV needs columns: {sorted(required)}")
        shots: list[dict[str, str | int]] = []
        seen: set[str] = set()
        for row in reader:
            shot_id = row["shot_id"].strip()
            cluster_id = row["cluster_id"].strip()
            if not shot_id or not cluster_id:
                raise ValueError("shot_id and cluster_id must not be empty")
            if shot_id in seen:
                raise ValueError(f"Duplicate shot_id: {shot_id}")
            seen.add(shot_id)
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
            if start_frame < 0 or end_frame <= start_frame:
                raise ValueError(
                    f"Invalid frame interval for shot {shot_id}: "
                    f"[{start_frame}, {end_frame})"
                )
            shots.append({
                "shot_id": shot_id,
                "cluster_id": cluster_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
            })
    shots.sort(key=lambda shot: int(shot["start_frame"]))
    for previous, current in zip(shots, shots[1:]):
        if int(previous["end_frame"]) > int(current["start_frame"]):
            raise ValueError(
                f"Overlapping shot intervals: "
                f"{previous['shot_id']} and {current['shot_id']}"
            )
    if not shots:
        raise ValueError("No shot clusters found")
    return shots


def load_camera_view_map(path: Path, clusters: set[str]) -> dict[str, str]:
    """Load human-reviewed labels; never infer meanings from numeric cluster IDs."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Camera-view map must be a JSON object keyed by cluster_id")
    mapping: dict[str, str] = {}
    for cluster_id in clusters:
        if cluster_id not in data:
            raise ValueError(
                f"Missing camera view for cluster {cluster_id}; "
                "use needs_review until its representative frames are checked"
            )
        label = data[cluster_id]
        if not isinstance(label, str) or label not in CAMERA_VIEWS:
            raise ValueError(f"Invalid camera view for cluster {cluster_id}: {label!r}")
        mapping[cluster_id] = label
    return mapping


def camera_view_template(shots: list[dict[str, str | int]]) -> dict[str, str]:
    """Give reviewers a placeholder for each existing visual cluster."""
    return {
        cluster_id: "needs_review"
        for cluster_id in sorted({str(shot["cluster_id"]) for shot in shots})
    }


def apply_camera_views(
    annotation_data: dict[str, object],
    shots: list[dict[str, str | int]],
    mapping: dict[str, str],
) -> dict[str, object]:
    """Decorate labelled frame records with shot, cluster, and reviewed view.

    Keep player boxes and masks untouched; require every frame to belong to a
    shot and never overwrite an already-reviewed, conflicting view.
    """
    frames = annotation_data.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Annotation JSON must contain a frames list")
    output_frames: list[dict[str, object]] = []
    for frame in frames:
        if not isinstance(frame, dict) or type(frame.get("frame_index")) is not int:
            raise ValueError("Each frame must contain an integer frame_index")
        index = frame["frame_index"]
        candidates = [
            shot for shot in shots
            if int(shot["start_frame"]) <= index < int(shot["end_frame"])
        ]
        if len(candidates) != 1:
            raise ValueError(f"Frame {index} has no unique shot-cluster assignment")
        shot = candidates[0]
        cluster_id = str(shot["cluster_id"])
        if cluster_id not in mapping:
            raise ValueError(f"Unmapped visual cluster: {cluster_id}")
        view = mapping[cluster_id]
        original_view = frame.get("camera_view")
        if original_view not in (None, "unknown", "needs_review", view):
            raise ValueError(
                f"Frame {index} already has camera_view={original_view!r}; "
                f"cluster proposes {view!r}. Review this shot manually."
            )
        output_frames.append({
            **frame,
            "shot_id": shot["shot_id"],
            "visual_cluster_id": cluster_id,
            "camera_view": view,
        })
    return {**annotation_data, "frames": output_frames}
