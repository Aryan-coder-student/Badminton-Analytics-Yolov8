"""Prepare a BFMD match for existing shot-cluster and player labelling flows.

Usage: uv run python scripts/prepare_bfmd.py --bfmd-root BFMD_data --match Antonsen \
    --transnet-weights /path/to/transnetv2-pytorch-weights.pth

For an already-segmented video, supply --shot-ranges path/to/shots.csv instead.
This script does not download or redistribute copyrighted BWF TV recordings.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

from badminton_analysis.evaluation.bfmd import (
    court_crop_bounds,
    find_bfmd_video,
    nearby_court_hint,
    read_bfmd_pose_prelabels,
    read_court_hints,
    save_contact_sheets,
    scenes_to_shots,
    select_cluster_frames,
)
from badminton_analysis.evaluation.camera_views import (
    camera_view_template,
    load_camera_view_map,
    load_shot_view_overrides,
)
from badminton_analysis.evaluation.shot_clustering import (
    extract_shot_embeddings,
    group_shots,
    load_embedding_cache,
    load_shot_ranges,
    save_embedding_cache,
    write_shot_clusters,
)


def _save_shots(path: Path, shots: list[object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("shot_id", "start_frame", "end_frame"))
        for shot in shots:
            writer.writerow((shot.shot_id, shot.start_frame, shot.end_frame))


def _detect_shots(video: Path, weights: Path, count: int, threshold: float) -> list:
    if not weights.is_file():
        raise FileNotFoundError(f"TransNetV2 weights not found: {weights}")
    import torch
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2(device="cpu")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        scenes = model.detect_scenes(str(video), threshold=threshold)
    return scenes_to_shots(scenes, count)


def _read_video_spec(video: Path) -> tuple[int, int, int, float]:
    reader = cv2.VideoCapture(str(video))
    if not reader.isOpened():
        raise ValueError(f"Cannot open BFMD video: {video}")
    try:
        count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(reader.get(cv2.CAP_PROP_FPS))
    finally:
        reader.release()
    if not all((count, width, height)) or fps <= 0:
        raise ValueError(f"Invalid video metadata: {video}")
    return count, width, height, fps


def run(args: argparse.Namespace) -> dict[str, object]:
    root = args.bfmd_root.expanduser()
    if args.video is None:
        video, match, kind = find_bfmd_video(root, args.match)
    else:
        video = args.video.expanduser()
        if not video.is_file():
            raise FileNotFoundError(video)
        match = video.stem
        kind = "doubles" if "videos_doubles" in video.parts else "singles"
    count, width, height, fps = _read_video_spec(video)
    if (
        (width, height) != (1280, 720) or abs(fps - 30) > 0.05
    ) and not args.allow_video_mismatch:
        raise ValueError(
            f"BFMD annotations expect 1280x720 at 30fps, found {width}x{height} "
            f"at {fps:.4f}fps. Use the official BFMD downloader to preserve "
            "frame alignment, or --allow-video-mismatch for exploratory "
            "clustering WITHOUT using BFMD's pixel/frame annotations."
        )
    use_annotations = (width, height) == (1280, 720) and abs(fps - 30) <= 0.05
    out = args.output or Path("data/benchmark/bfmd") / match
    out.mkdir(parents=True, exist_ok=True)
    shot_csv = out / "shot_ranges.csv"
    if args.shot_ranges is not None:
        shots = load_shot_ranges(args.shot_ranges)
    elif shot_csv.exists() and not args.redetect:
        shots = load_shot_ranges(shot_csv)
    else:
        if args.transnet_weights is None:
            raise ValueError(
                "Supply --transnet-weights to run TransNetV2, or "
                "--shot-ranges with existing TransNetV2 shot boundaries."
            )
        shots = _detect_shots(video, args.transnet_weights, count, args.threshold)
    if any(shot.end_frame > count for shot in shots):
        raise ValueError("Shot bounds exceed source video length")
    if args.max_shots is not None:
        if args.max_shots < 2:
            raise ValueError("--max-shots must be at least 2")
        shots = shots[: args.max_shots]
    _save_shots(shot_csv, shots)

    cache_path = out / "shot_embeddings.npz"
    if cache_path.is_file() and not args.reembed:
        cached_shots, features, sample_indices = load_embedding_cache(cache_path)
        if cached_shots != shots:
            raise ValueError(
                "Cached shot ranges differ from requested video. Use --reembed "
                "or specify a fresh --output directory."
            )
    else:
        from badminton_analysis.model.embeddings import Embedding

        features, sample_indices = extract_shot_embeddings(
            video, shots, Embedding("dinov2")
        )
        save_embedding_cache(cache_path, shots, features, sample_indices)
    labels, silhouette = group_shots(features, args.clusters)
    cluster_csv = out / "shot_clusters.csv"
    write_shot_clusters(cluster_csv, shots, labels, sample_indices)
    camera_map = out / "cluster_camera_views.json"
    if not camera_map.exists():
        camera_map.write_text(
            json.dumps(camera_view_template([
                {"cluster_id": int(label)} for label in labels
            ]), indent=2) + "\n",
            encoding="utf-8",
        )
    cluster_ids = {str(label) for label in labels}
    view_mapping = load_camera_view_map(camera_map, cluster_ids)
    override_file = out / "shot_view_overrides.json"
    view_overrides = (
        load_shot_view_overrides(
            override_file, {shot.shot_id for shot in shots}
        ) if override_file.exists() else {}
    )

    selected = select_cluster_frames(
        shots, labels, sample_indices, target=args.frames, seed=args.seed
    )
    indices = {int(sample["frame_index"]) for sample in selected}
    # Hold out entire shots, not nearby frames, to avoid evaluation leakage.
    shot_ids = sorted({str(item["shot_id"]) for item in selected})
    shuffled = list(shot_ids)
    random.Random(args.seed).shuffle(shuffled)
    holdout_count = max(1, round(len(shuffled) * 0.2)) if len(shuffled) > 1 else 0
    holdout_shots = set(shuffled[:holdout_count])
    court_path = root / "annotations" / "court_perframe" / f"{match}.json"
    court_hints = read_court_hints(court_path) if use_annotations else []
    pose_path = root / "annotations" / "pose" / f"{match}.json"
    pose_prelabels = (
        read_bfmd_pose_prelabels(pose_path, indices) if use_annotations else {}
    )
    images = out / "images"
    rois = out / "court_roi_previews"
    images.mkdir(exist_ok=True)
    rois.mkdir(exist_ok=True)
    reader = cv2.VideoCapture(str(video))
    records = []
    annotations = []
    try:
        for sample in selected:
            index = int(sample["frame_index"])
            if not reader.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise ValueError(f"Cannot seek to BFMD frame {index}")
            ok, frame = reader.read()
            if not ok:
                raise ValueError(f"Cannot decode BFMD frame {index}")
            frame_name = f"frame_{index:08d}.png"
            image_path = images / frame_name
            if not cv2.imwrite(str(image_path), frame):
                raise ValueError(f"Cannot save frame {index}")
            hint = nearby_court_hint(court_hints, index)
            bounds = court_crop_bounds(hint, width, height)
            roi_path = ""
            if bounds:
                x1, y1, x2, y2 = bounds
                crop_path = rois / frame_name
                if not cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2]):
                    raise ValueError(f"Cannot save court preview {index}")
                roi_path = str(crop_path.relative_to(out))
            shot_id = str(sample["shot_id"])
            view = view_overrides.get(
                shot_id, view_mapping[str(sample["cluster_id"])]
            )
            view_source = (
                "shot_override" if shot_id in view_overrides
                else "reviewed_cluster"
            )
            item = {
                **sample,
                "split": "holdout" if shot_id in holdout_shots else "development",
                "camera_view": view,
                "camera_view_source": view_source,
                "image_path": str(image_path.relative_to(out)),
                "court_roi_preview": roi_path,
                "court_flag_hint": str(hint.get("flag")) if hint else "unknown",
                "court_hint_frame": int(hint["frame_idx"]) if hint else None,
                "annotation_status": (
                    "unreviewed_prelabel" if index in pose_prelabels
                    else "unlabelled"
                ),
            }
            records.append(item)
            annotations.append({
                "frame_index": index,
                "shot_id": sample["shot_id"],
                "visual_cluster_id": str(sample["cluster_id"]),
                "camera_view": view,
                "camera_view_source": view_source,
                "split": item["split"],
                "annotation_status": item["annotation_status"],
                "players": pose_prelabels.get(index, []),
            })
    finally:
        reader.release()
    output_csv = out / "candidate_frames.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (out / "annotations_PRELABELS_NOT_GROUND_TRUTH.json").write_text(
        json.dumps({
            "video": str(video),
            "source": "BFMD pose bounding boxes where available; NOT manually verified",
            "frames": annotations,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    save_contact_sheets(selected, images, out / "cluster_review")
    report = {
        "dataset": "BFMD",
        "match": match,
        "match_type": kind,
        "video": str(video),
        "video_frames": count,
        "fps": fps,
        "total_shots": len(shots),
        "visual_clusters": len(set(int(label) for label in labels)),
        "silhouette": silhouette,
        "exported_frames": len(records),
        "prelabelled_frames": len(pose_prelabels),
        "court_hint_frames_available": bool(court_hints),
        "camera_views_verified": all(
            item["camera_view"] != "needs_review" for item in records
        ),
        "holdout_frames": sum(item["split"] == "holdout" for item in records),
        "player_ground_truth_verified": False,
        "reminder": (
            "Cluster IDs are visual groups, not camera labels. BFMD pose boxes "
            "are unverified prelabels. Review boxes and draw masks in an "
            "annotation tool before evaluating detection accuracy. "
            "Court ROI images are previews, NOT input to shot-view embeddings."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfmd-root", type=Path, required=True)
    parser.add_argument("--match", required=True, help="Unique match filename fragment")
    parser.add_argument("--video", type=Path, help="Override local video discovery")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shot-ranges", type=Path,
                        help="Reuse existing TransNetV2 shot boundaries CSV")
    parser.add_argument("--transnet-weights", type=Path,
                        help="TransNetV2 PyTorch state-dict checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-shots", type=int,
                        help="Smoke test on the earliest N shots only")
    parser.add_argument("--redetect", action="store_true")
    parser.add_argument("--reembed", action="store_true")
    parser.add_argument("--allow-video-mismatch", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
