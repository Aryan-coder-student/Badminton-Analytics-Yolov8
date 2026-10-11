"""Convert human-reviewed CVAT COCO 1.0 annotations into benchmark GT.

The --confirm-reviewed CLI flag is an explicit human declaration; no program
can automatically certify that each annotation was manually verified.
"""

import json
from pathlib import Path
import re
from zipfile import ZipFile

import cv2
import numpy as np


FRAME_NAME = re.compile(r"frame_(\d{8,})\.png$", flags=re.IGNORECASE)


def _load_coco(path: Path) -> dict[str, object]:
    if path.suffix.lower() != ".zip":
        result = json.loads(path.read_text(encoding="utf-8"))
    else:
        with ZipFile(path) as archive:
            names = [
                name for name in archive.namelist()
                if name.startswith("annotations/") and name.endswith(".json")
            ]
            if len(names) != 1:
                raise ValueError(
                    "Expected exactly one COCO JSON under annotations/ in CVAT ZIP"
                )
            result = json.loads(archive.read(names[0]))
    if not isinstance(result, dict):
        raise ValueError("COCO annotations must be an object")
    if not all(isinstance(result.get(key), list) for key in (
        "images", "annotations", "categories"
    )):
        raise ValueError("COCO JSON must contain images, annotations and categories")
    return result


def convert_reviewed_cvat(
    reviewed_coco: Path,
    source_manifest: Path,
    output_manifest: Path,
    *,
    confirmed_reviewed: bool,
) -> dict[str, int]:
    """Import a CVAT-reviewed COCO dataset into exact-pixel GT masks/boxes."""
    if not confirmed_reviewed:
        raise ValueError("Require explicit confirmation that all frames were reviewed")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    originals = source.get("frames")
    if not isinstance(originals, list) or not originals:
        raise ValueError("Source BFMD manifest must contain selected frames")
    coco = _load_coco(reviewed_coco)
    categories = {
        item["id"]: item["name"] for item in coco["categories"]
    }
    if not categories or any(name != "player" for name in categories.values()):
        raise ValueError(
            "Expected reviewed COCO to contain only category 'player'; "
            "remove non-player categories before export"
        )
    images: dict[int, tuple[int, int, int]] = {}
    for item in coco["images"]:
        name = Path(item["file_name"]).name
        found = FRAME_NAME.fullmatch(name)
        if found is None:
            raise ValueError(f"Unexpected CVAT image name: {name}")
        index = int(found.group(1))
        if index in (frame[0] for frame in images.values()):
            raise ValueError(f"Duplicate CVAT frame index: {index}")
        width, height = int(item["width"]), int(item["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid CVAT image dimensions for frame {index}")
        images[int(item["id"])] = (index, width, height)
    expected = {frame["frame_index"] for frame in originals}
    actual = {frame_index for frame_index, _, _ in images.values()}
    if expected != actual:
        raise ValueError(
            f"CVAT images differ from selected BFMD frames: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    players: dict[int, list[dict[str, object]]] = {
        index: [] for index in expected
    }
    mask_dir = output_manifest.parent / (output_manifest.stem + "_masks")
    mask_dir.mkdir(parents=True, exist_ok=True)
    for item in coco["annotations"]:
        image_id = item["image_id"]
        if image_id not in images:
            raise ValueError(f"Annotation points to unknown COCO image {image_id}")
        if item["category_id"] not in categories:
            raise ValueError("Unknown COCO player category_id")
        frame_index, width, height = images[image_id]
        coords = item["bbox"]
        if len(coords) != 4 or not all(np.isfinite(float(v)) for v in coords):
            raise ValueError(f"Invalid COCO bbox on frame {frame_index}")
        x, y, w, h = [float(v) for v in coords]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ValueError(f"Out-of-frame bbox on frame {frame_index}")
        label: dict[str, object] = {"bbox": [x, y, x + w, y + h]}
        segmentation = item.get("segmentation", [])
        if segmentation:
            if isinstance(segmentation, dict):
                # CVAT COCO exports genuine masks as compressed or uncompressed
                # COCO RLE. Decode at native resolution; do not polygonize.
                from pycocotools import mask as mask_utils

                rle = segmentation
                if isinstance(rle.get("counts"), list):
                    rle = mask_utils.frPyObjects(rle, height, width)
                mask = np.asarray(mask_utils.decode(rle), dtype=np.uint8)
                if mask.shape != (height, width):
                    raise ValueError("COCO RLE shape differs from source frame")
                mask = (mask > 0).astype(np.uint8) * 255
            elif isinstance(segmentation, list):
                mask = np.zeros((height, width), dtype=np.uint8)
                valid_polygons = []
                for polygon in segmentation:
                    if len(polygon) < 6 or len(polygon) % 2:
                        raise ValueError("Malformed COCO player polygon")
                    vertices = np.asarray(polygon, dtype=float).reshape(-1, 2)
                    if not np.isfinite(vertices).all():
                        raise ValueError("Non-finite COCO polygon")
                    valid_polygons.append(np.rint(vertices).astype(np.int32))
                cv2.fillPoly(mask, valid_polygons, 255)
            else:
                raise ValueError("Unsupported COCO segmentation encoding")
            if not np.any(mask):
                raise ValueError("Empty COCO player mask")
            name = f"frame_{frame_index:08d}_player_{len(players[frame_index]):02d}.png"
            mask_file = mask_dir / name
            if not cv2.imwrite(str(mask_file), mask):
                raise ValueError(f"Cannot write ground truth mask {mask_file}")
            label["mask_path"] = f"{mask_dir.name}/{name}"
        players[frame_index].append(label)
    output_frames = [
        {
            **frame,
            "annotation_status": "human_reviewed",
            "players": players[frame["frame_index"]],
        }
        for frame in originals
    ]
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps({
            **source,
            "stage": "reviewed_cvat_ground_truth",
            "frames": output_frames,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "frames": len(output_frames),
        "players": sum(map(len, players.values())),
        "masks": sum(
            "mask_path" in player
            for labels in players.values() for player in labels
        ),
    }
