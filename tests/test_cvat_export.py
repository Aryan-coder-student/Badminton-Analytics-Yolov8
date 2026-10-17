"""Test CVAT COCO prelabel export without pretrained models."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

import cv2
import numpy as np

from badminton_analysis.evaluation.cvat import (
    create_cvat_prelabels,
    mask_polygons,
)


class CvatExportTests(unittest.TestCase):
    def test_polygon_generation_for_square_mask(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[3:10, 4:12] = 255
        polys = mask_polygons(mask)
        self.assertEqual(len(polys), 1)
        self.assertGreaterEqual(len(polys[0]), 8)

    def test_complete_package_and_unlabelled_images(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            for index in (17, 22):
                cv2.imwrite(
                    str(images / f"frame_{index:08d}.png"),
                    np.zeros((30, 40, 3), dtype=np.uint8),
                )
            mask_dir = root / "yolo_predictions_masks"
            mask_dir.mkdir()
            mask = np.zeros((30, 40), dtype=np.uint8)
            mask[5:21, 10:22] = 255
            cv2.imwrite(str(mask_dir / "person.png"), mask)
            source = root / "annotations_PRELABELS_NOT_GROUND_TRUTH.json"
            source.write_text(json.dumps({"frames": [
                {"frame_index": 17, "players": []},
                {"frame_index": 22, "players": []},
            ]}))
            predictions = root / "yolo_predictions.json"
            predictions.write_text(json.dumps({"frames": [
                {"frame_index": 17, "players": [{
                    "bbox": [10, 5, 22, 21],
                    "mask_path": "yolo_predictions_masks/person.png",
                }]},
                {"frame_index": 22, "players": []},
            ]}))
            archive = root / "cvat.zip"
            summary = create_cvat_prelabels(source, predictions, images, archive)
            self.assertEqual(summary["images"], 2)
            self.assertEqual(summary["prelabel_instances"], 1)
            with ZipFile(archive) as file:
                self.assertIn("annotations/instances_default.json", file.namelist())
                self.assertIn("images/frame_00000017.png", file.namelist())
                self.assertIn("images/frame_00000022.png", file.namelist())
                self.assertIn("ANNOTATION_STATUS.txt", file.namelist())
                coco = json.loads(
                    file.read("annotations/instances_default.json")
                )
                self.assertEqual(len(coco["images"]), 2)
                self.assertEqual(len(coco["annotations"]), 1)
                self.assertEqual(coco["annotations"][0]["category_id"], 1)
                self.assertTrue(coco["annotations"][0]["segmentation"])

    def test_unknown_prediction_frame_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "source.json"
            pred = root / "predictions.json"
            src.write_text(json.dumps({"frames": [
                {"frame_index": 1, "players": []}
            ]}))
            pred.write_text(json.dumps({"frames": [
                {"frame_index": 2, "players": []}
            ]}))
            with self.assertRaisesRegex(ValueError, "unexpected frames"):
                create_cvat_prelabels(src, pred, root, root / "out.zip")

    def test_unaligned_mask_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            cv2.imwrite(
                str(root / "images" / "frame_00000001.png"),
                np.zeros((30, 40, 3), dtype=np.uint8),
            )
            cv2.imwrite(
                str(root / "mask.png"),
                np.full((15, 20), 255, dtype=np.uint8),
            )
            src = root / "source.json"
            pred = root / "predictions.json"
            src.write_text(json.dumps({"frames": [
                {"frame_index": 1, "players": []}
            ]}))
            pred.write_text(json.dumps({"frames": [
                {"frame_index": 1, "players": [{
                    "bbox": [2, 3, 10, 15],
                    "mask_path": "mask.png",
                }]}
            ]}))
            with self.assertRaisesRegex(ValueError, "not aligned"):
                create_cvat_prelabels(src, pred, root / "images", root / "bad.zip")


if __name__ == "__main__":
    unittest.main()
