"""Export machine-generated player masks for human review in CVAT (COCO 1.0).

The resulting annotations are predictions, not verified ground truth. Human
review must remove off-court people, add missed players, and correct masks.
"""

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np


def mask_polygons(mask: np.ndarray) -> list[list[float]]:
    """Convert outer mask contours to COCO polygon candidates.

    COCO polygons cannot encode mask holes exactly. The result is for CVAT
    manual correction, never the source of benchmark ground truth.
    """
    if mask.ndim != 2:
        raise ValueError("Expected a single-channel instance mask")
    binary = np.asarray(mask > 0, dtype=np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) <= 0:
            continue
        polygon = [float(v) for v in contour.reshape(-1)]
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons


def create_cvat_prelabels(
    source_manifest: Path,
    predictions_file: Path,
    images_dir: Path,
    zip_path: Path,
) -> dict[str, int]:
    """Pack all selected frames and matching YOLO masks into a CVAT COCO ZIP."""
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_file.read_text(encoding="utf-8"))
    source_frames = source.get("frames")
    predicted_frames = predictions.get("frames")
    if not isinstance(source_frames, list) or not isinstance(predicted_frames, list):
        raise ValueError("Both input manifests must contain a frames list")
    if not source_frames:
        raise ValueError("No candidate BFMD frames found")
    predicted_by_frame: dict[int, list[dict[str, object]]] = {}
    for frame in predicted_frames:
        index = frame["frame_index"]
        if type(index) is not int or index in predicted_by_frame:
            raise ValueError(f"Duplicate or invalid predicted frame: {index!r}")
        predicted_by_frame[index] = frame["players"]

    known_indices = [frame["frame_index"] for frame in source_frames]
    if any(type(i) is not int for i in known_indices):
        raise ValueError("Frame indices must be integers")
    if len(set(known_indices)) != len(known_indices):
        raise ValueError("Duplicate source frames")
    unknown = set(predicted_by_frame) - set(known_indices)
    if unknown:
        raise ValueError(f"Predictions include unexpected frames: {sorted(unknown)}")

    coco: dict[str, object] = {
        "info": {
            "description": (
                "UNVERIFIED YOLO person pre-annotations; correct all boxes and "
                "masks before treating them as player ground truth"
            ),
            "version": "0.1",
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "player", "supercategory": "person"}],
        "images": [],
        "annotations": [],
    }
    image_paths: list[tuple[Path, str]] = []
    next_id = 1
    for image_id, frame in enumerate(source_frames, start=1):
        frame_index = frame["frame_index"]
        filename = f"frame_{frame_index:08d}.png"
        path = images_dir / filename
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read source frame: {path}")
        height, width = image.shape[:2]
        image_paths.append((path, filename))
        coco["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
            "source_frame_index": frame_index,
        })
        for prediction in predicted_by_frame.get(frame_index, []):
            bbox = prediction["bbox"]
            if len(bbox) != 4 or not all(np.isfinite(float(v)) for v in bbox):
                raise ValueError(f"Invalid predicted bbox for frame {frame_index}")
            x1, y1, x2, y2 = [float(value) for value in bbox]
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            mask_path = prediction.get("mask_path")
            segmentation: list[list[float]] = []
            area = (x2 - x1) * (y2 - y1)
            if mask_path:
                relative = Path(str(mask_path))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Prediction mask path must be relative to JSON")
                mask = cv2.imread(
                    str(predictions_file.parent / relative), cv2.IMREAD_GRAYSCALE
                )
                if mask is None or mask.shape != (height, width):
                    raise ValueError(
                        f"Prediction mask missing or not aligned: {relative}"
                    )
                segmentation = mask_polygons(mask)
                if segmentation:
                    area = float(np.count_nonzero(mask))
            coco["annotations"].append({
                "id": next_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": segmentation,
                "area": area,
                "iscrowd": 0,
            })
            next_id += 1
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", ZIP_DEFLATED, compresslevel=3) as archive:
        archive.writestr(
            "annotations/instances_default.json",
            json.dumps(coco, separators=(",", ":")),
        )
        archive.writestr(
            "ANNOTATION_STATUS.txt",
            "UNVERIFIED PREDICTIONS, NOT GROUND TRUTH.\n"
            "Review every frame in CVAT. Correct masks/boxes, remove spectators, "
            "add missing players, and export the corrected dataset for evaluation.\n",
        )
        for path, filename in image_paths:
            archive.write(path, f"images/{filename}")
    return {
        "images": len(coco["images"]),
        "prelabel_instances": len(coco["annotations"]),
        "frames_with_predictions": len(predicted_by_frame),
    }
