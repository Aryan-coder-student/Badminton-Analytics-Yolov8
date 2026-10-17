"""BFMD dataset adapters: video discovery, scene bounds, court hints, candidate selection.

BFMD annotations and video are separate downloads. Frames are never ground truth
until their player boxes/masks have been reviewed by an annotator.
"""

from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
import json
import random

import cv2
import numpy as np

from badminton_analysis.evaluation.shot_clustering import ShotRange


def find_bfmd_video(root: Path, match_query: str) -> tuple[Path, str, str]:
    """Find exactly one local BFMD match video by case-insensitive name fragment."""
    if not root.is_dir():
        raise FileNotFoundError(f"BFMD_data folder missing: {root}")
    candidates = [
        path
        for folder in ("videos", "videos_doubles")
        for path in (root / folder).rglob("*.mp4")
        if match_query.casefold() in path.stem.casefold()
    ]
    if len(candidates) != 1:
        names = "\n".join(str(path.relative_to(root)) for path in candidates[:20])
        raise ValueError(
            f"Expected one local BFMD video for {match_query!r}; "
            f"found {len(candidates)}. {names}\n"
            "Use BFMD_data/download_youtube.py --list and --match to obtain "
            "source footage separately, subject to BWF TV rights."
        )
    path = candidates[0]
    kind = "doubles" if "videos_doubles" in path.parts else "singles"
    return path, path.stem, kind


def scenes_to_shots(
    scenes: list[dict[str, object]],
    frame_count: int,
) -> list[ShotRange]:
    """Convert TransNetV2 inclusive scene-end frames to exclusive shot ranges."""
    if frame_count <= 0:
        raise ValueError("Source video has no readable frames")
    shots: list[ShotRange] = []
    for index, scene in enumerate(scenes, 1):
        start = int(scene["start_frame"])
        end_exclusive = int(scene["end_frame"]) + 1
        if start < 0 or not start < end_exclusive <= frame_count:
            raise ValueError(f"Invalid TransNetV2 scene bounds: {scene}")
        shots.append(ShotRange(str(scene.get("shot_id", index)), start, end_exclusive))
    if not shots:
        raise ValueError("TransNetV2 did not return any camera shots")
    shots.sort(key=lambda shot: shot.start_frame)
    if len({shot.shot_id for shot in shots}) != len(shots):
        raise ValueError("TransNetV2 returned duplicate shot IDs")
    for previous, current in zip(shots, shots[1:]):
        if previous.end_frame > current.start_frame:
            raise ValueError("TransNetV2 shot intervals overlap")
    return shots


def read_court_hints(path: Path) -> list[dict[str, object]]:
    """Read BFMD court_perframe observations; annotations may be absent for doubles."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames", []) if isinstance(data, dict) else []
    if not isinstance(frames, list):
        raise ValueError("BFMD court_perframe.frames must be a list")
    result = []
    for frame in frames:
        if not isinstance(frame, dict) or "frame_idx" not in frame:
            continue
        flag = frame.get("flag")
        if flag not in {"valid", "refit", "no_court"}:
            continue
        result.append(frame)
    result.sort(key=lambda frame: int(frame["frame_idx"]))
    return result


def nearby_court_hint(
    hints: list[dict[str, object]],
    frame_index: int,
    max_gap: int = 15,
) -> dict[str, object] | None:
    """Find a time-nearby court observation without extrapolating over cuts."""
    if not hints:
        return None
    positions = [int(frame["frame_idx"]) for frame in hints]
    index = bisect_left(positions, frame_index)
    options = [hints[i] for i in (index - 1, index) if 0 <= i < len(hints)]
    nearest = min(options, key=lambda frame: abs(int(frame["frame_idx"]) - frame_index))
    if abs(int(nearest["frame_idx"]) - frame_index) > max_gap:
        return None
    return nearest


def court_crop_bounds(
    hint: dict[str, object] | None,
    width: int,
    height: int,
    padding_ratio: float = 0.15,
) -> tuple[int, int, int, int] | None:
    """Return a preview ROI; never use this crop for shot-view embeddings."""
    if hint is None or hint.get("flag") not in {"valid", "refit"}:
        return None
    corners = np.asarray(hint.get("corners"), dtype=float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return None
    minx, miny = np.min(corners, axis=0)
    maxx, maxy = np.max(corners, axis=0)
    if maxx <= minx or maxy <= miny:
        return None
    pad_x, pad_y = (maxx - minx) * padding_ratio, (maxy - miny) * padding_ratio
    x1 = max(0, int(np.floor(minx - pad_x)))
    y1 = max(0, int(np.floor(miny - pad_y)))
    x2 = min(width, int(np.ceil(maxx + pad_x)))
    y2 = min(height, int(np.ceil(maxy + pad_y)))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def select_cluster_frames(
    shots: list[ShotRange],
    labels: np.ndarray,
    sample_indices: list[tuple[int, ...]],
    target: int = 150,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Distribute reviewed examples across clusters and time, then shot interiors.

    Select only one frame per shot in the first round to reduce near-duplicates.
    The second round adds other 25/50/75%-sampled frames when needed.
    """
    if target < 1:
        raise ValueError("target must be positive")
    if len(shots) != len(labels) or len(shots) != len(sample_indices):
        raise ValueError("Shot sample counts do not agree")
    groups: dict[int, list[tuple[ShotRange, tuple[int, ...]]]] = defaultdict(list)
    for shot, cluster, indices in zip(shots, labels, sample_indices, strict=True):
        if not indices:
            raise ValueError(f"Shot {shot.shot_id} has no sampled frames")
        groups[int(cluster)].append((shot, indices))
    rng = random.Random(seed)
    for entries in groups.values():
        entries.sort(key=lambda item: item[0].start_frame)
        # Interleave random chunks of the ordered match for temporal coverage.
        chunks = [entries[i::10] for i in range(10)]
        rng.shuffle(chunks)
        entries[:] = [entry for chunk in chunks for entry in chunk]
    first_pass: dict[int, list[dict[str, object]]] = {}
    second_pass: dict[int, list[dict[str, object]]] = {}
    for group, entries in groups.items():
        first, extra = [], []
        for shot, indices in entries:
            base = {"shot_id": shot.shot_id, "cluster_id": group}
            middle = indices[len(indices) // 2]
            first.append({**base, "frame_index": middle})
            extra.extend(
                {**base, "frame_index": index}
                for index in indices if index != middle
            )
        first_pass[group] = first
        second_pass[group] = extra
    selected: list[dict[str, object]] = []
    # Exhaust different shots before selecting a second frame from any shot.
    for pools in (first_pass, second_pass):
        offsets = {group: 0 for group in pools}
        groups_in_order = sorted(pools)
        while len(selected) < target:
            progressed = False
            for group in groups_in_order:
                if offsets[group] < len(pools[group]):
                    selected.append(pools[group][offsets[group]])
                    offsets[group] += 1
                    progressed = True
                    if len(selected) == target:
                        break
            if not progressed:
                break
    selected.sort(key=lambda item: int(item["frame_index"]))
    return selected


def read_bfmd_pose_prelabels(
    path: Path,
    frame_indices: set[int],
) -> dict[int, list[dict[str, object]]]:
    """Convert BFMD pixel-coordinate pose boxes to unverified player prelabels."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, list[dict[str, object]]] = {}
    for rally in data.get("videos", {}).values():
        for frame_index, people in rally.get("frames", {}).items():
            index = int(frame_index)
            if index not in frame_indices:
                continue
            boxes = []
            for person in people:
                box = person.get("bbox")
                if box is None or len(box) != 4:
                    continue
                if not all(np.isfinite(float(v)) for v in box):
                    continue
                x1, y1, x2, y2 = map(float, box)
                if x2 > x1 and y2 > y1:
                    boxes.append({"bbox": [x1, y1, x2, y2]})
            if boxes:
                result[index] = boxes
    return result


def save_contact_sheets(
    samples: list[dict[str, object]],
    image_directory: Path,
    output_directory: Path,
    columns: int = 4,
) -> None:
    """Generate visual review sheets grouped by unsupervised cluster."""
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for item in samples:
        groups[int(item["cluster_id"])].append(item)
    output_directory.mkdir(parents=True, exist_ok=True)
    for cluster_id, group in groups.items():
        tile_w, tile_h = 320, 204
        rows = (len(group) + columns - 1) // columns
        canvas = np.full((rows * tile_h, columns * tile_w, 3), 240, dtype=np.uint8)
        for n, sample in enumerate(group):
            filename = f"frame_{int(sample['frame_index']):08d}.png"
            frame = cv2.imread(str(image_directory / filename))
            if frame is None:
                raise ValueError(f"Missing extracted image for review: {filename}")
            resized = cv2.resize(frame, (tile_w, tile_h - 26))
            r, c = divmod(n, columns)
            y, x = r * tile_h, c * tile_w
            canvas[y:y + tile_h - 26, x:x + tile_w] = resized
            cv2.putText(
                canvas,
                f"shot {sample['shot_id']} | frame {sample['frame_index']}",
                (x + 8, y + tile_h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1,
            )
        out = output_directory / f"cluster_{cluster_id:02d}.jpg"
        if not cv2.imwrite(str(out), canvas):
            raise ValueError(f"Failed to write contact sheet: {out}")
