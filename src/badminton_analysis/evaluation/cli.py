"""Generate YOLO player masks for labelled frames and evaluate against labels."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from badminton_analysis.evaluation.players import evaluate_frames, load_frames


def _predict(args: argparse.Namespace) -> None:
    # Import heavy model dependencies only when the user requests inference.
    from ultralytics import YOLO

    from badminton_analysis.model.segmentations import Segmenter
    from badminton_analysis.model.settings import get_settings

    annotations = load_frames(args.annotations)
    settings = get_settings()
    segmenter = Segmenter("yolo", options={
        "model": YOLO(args.model or settings.segmentation_model_name),
        "confidence_threshold": (
            args.confidence
            if args.confidence is not None
            else settings.segmentation_confidence
        ),
        # Filter COCO person class *before* mask upscaling.
        "class_ids": (0,),
        "device": args.device if args.device is not None else settings.segmentation_device,
    })

    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        raise ValueError(f"Cannot open video: {args.video}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output.parent / (args.output.stem + "_masks")
    mask_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    try:
        for frame in sorted(annotations, key=lambda item: item.frame_index):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame.frame_index)
            ok, image = video.read()
            if not ok:
                raise ValueError(f"Cannot read video frame: {frame.frame_index}")
            candidates = segmenter.segment(image)
            players: list[dict[str, object]] = []
            for index, candidate in enumerate(candidates):
                if candidate.mask.shape != image.shape[:2]:
                    raise ValueError(
                        f"Segmentation mask has wrong shape at frame {frame.frame_index}"
                    )
                mask_name = f"frame_{frame.frame_index:08d}_person_{index:02d}.png"
                mask_path = mask_dir / mask_name
                if not cv2.imwrite(str(mask_path), candidate.mask.astype(np.uint8) * 255):
                    raise ValueError(f"Cannot write predicted mask: {mask_path}")
                players.append({
                    "bbox": list(candidate.detection.bbox),
                    "confidence": candidate.detection.confidence,
                    "mask_path": f"{mask_dir.name}/{mask_name}",
                })
            records.append({"frame_index": frame.frame_index, "players": players})
    finally:
        video.release()
    args.output.write_text(json.dumps({
        "stage": "raw_yolo_person_segmentation",
        "video": str(args.video),
        "frames": records,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} frames of raw predictions to {args.output}")


def _evaluate(args: argparse.Namespace) -> None:
    report = evaluate_frames(
        load_frames(args.annotations),
        load_frames(args.predictions),
        iou_threshold=args.iou_threshold,
    )
    report_json = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json, encoding="utf-8")
        print(f"Saved evaluation report to {args.output}")
    else:
        print(report_json, end="")



def _camera_view_template(args: argparse.Namespace) -> None:
    from badminton_analysis.evaluation.camera_views import (
        camera_view_template,
        load_shot_clusters,
    )

    template = camera_view_template(load_shot_clusters(args.shot_clusters))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(template, indent=2) + "\\n", encoding="utf-8")
    print(f"Review the camera views for {len(template)} visual clusters: {args.output}")


def _apply_camera_views(args: argparse.Namespace) -> None:
    from badminton_analysis.evaluation.camera_views import (
        apply_camera_views,
        load_camera_view_map,
        load_shot_clusters,
    )

    # Masks are stored relative to the annotation manifest directory.
    if args.output.resolve().parent != args.annotations.resolve().parent:
        raise ValueError("Keep output JSON beside annotations to preserve relative mask paths")
    load_frames(args.annotations)
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    shots = load_shot_clusters(args.shot_clusters)
    cluster_ids = {str(shot["cluster_id"]) for shot in shots}
    mapping = load_camera_view_map(args.cluster_map, cluster_ids)
    enriched = apply_camera_views(annotations, shots, mapping)
    args.output.write_text(json.dumps(enriched, indent=2) + "\\n", encoding="utf-8")
    print(f"Tagged {len(enriched['frames'])} frames using reviewed shot clusters")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    predict = subcommands.add_parser(
        "predict", help="Run YOLO person segmentation on annotated video frames"
    )
    predict.add_argument("--video", type=Path, required=True)
    predict.add_argument("--annotations", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument(
        "--model", help="YOLO segmentation checkpoint; defaults to project settings"
    )
    predict.add_argument(
        "--confidence", type=float, help="Confidence; defaults to project settings"
    )
    predict.add_argument("--device", help="Inference device such as cpu, cuda:0, or mps")
    predict.set_defaults(handler=_predict)

    evaluate = subcommands.add_parser(
        "evaluate", help="Measure detection and mask accuracy on annotated frames"
    )
    evaluate.add_argument("--annotations", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--iou-threshold", type=float, default=0.5)
    evaluate.set_defaults(handler=_evaluate)

    template = subcommands.add_parser(
        "camera-view-template",
        help="Create a needs_review map for existing shot cluster IDs",
    )
    template.add_argument("--shot-clusters", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    template.set_defaults(handler=_camera_view_template)

    apply_views = subcommands.add_parser(
        "camera-views",
        help="Attach reviewed shot-cluster camera views to existing annotations",
    )
    apply_views.add_argument("--shot-clusters", type=Path, required=True)
    apply_views.add_argument("--cluster-map", type=Path, required=True)
    apply_views.add_argument("--annotations", type=Path, required=True)
    apply_views.add_argument("--output", type=Path, required=True)
    apply_views.set_defaults(handler=_apply_camera_views)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
