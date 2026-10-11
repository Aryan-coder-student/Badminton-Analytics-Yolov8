"""Review-only COCO-to-ground-truth conversion tests."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2

from badminton_analysis.evaluation.cvat_import import convert_reviewed_cvat
from badminton_analysis.evaluation.players import load_frames


class CvatImportTests(unittest.TestCase):
    def test_manual_confirmation_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "confirmation"):
            convert_reviewed_cvat(
                Path("not-read.json"), Path("not-read.json"),
                Path("not-written.json"), confirmed_reviewed=False,
            )

    def test_reviewed_polygons_generate_aligned_masks_and_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "prelabels.json"
            source.write_text(json.dumps({
                "frames": [
                    {"frame_index": 17, "players": [], "split": "development",
                     "camera_view": "wide", "shot_id": "a"},
                    {"frame_index": 22, "players": [], "split": "holdout",
                     "camera_view": "closeup", "shot_id": "b"},
                ]
            }))
            coco = root / "coco.json"
            coco.write_text(json.dumps({
                "images": [
                    {"id": 101, "file_name": "images/frame_00000017.png",
                     "width": 40, "height": 30},
                    {"id": 102, "file_name": "frame_00000022.png",
                     "width": 40, "height": 30},
                ],
                "categories": [{"id": 1, "name": "player"}],
                "annotations": [{
                    "id": 1, "image_id": 101, "category_id": 1,
                    "bbox": [8, 6, 13, 11], "iscrowd": 0,
                    "segmentation": [[8, 6, 21, 6, 21, 17, 8, 17]],
                }],
            }))
            output = root / "verified_ground_truth.json"
            report = convert_reviewed_cvat(
                coco, source, output, confirmed_reviewed=True,
            )
            self.assertEqual(report, {"frames": 2, "players": 1, "masks": 1})
            data = json.loads(output.read_text())
            self.assertEqual(data["frames"][1]["players"], [])
            self.assertEqual(data["frames"][1]["split"], "holdout")
            self.assertEqual(data["frames"][0]["shot_id"], "a")
            self.assertEqual(
                data["frames"][0]["annotation_status"], "human_reviewed",
            )
            frame_data = load_frames(output)
            self.assertEqual(frame_data[0].players[0].bbox, (8., 6., 21., 17.))
            mask = frame_data[0].players[0].mask
            self.assertEqual(mask.shape, (30, 40))
            self.assertTrue(mask[10, 12])
            self.assertFalse(mask[0, 0])

    def test_missing_selected_image_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "selected.json"
            manifest.write_text(json.dumps({
                "frames": [{"frame_index": 33, "players": []}]
            }))
            coco = root / "coco.json"
            coco.write_text(json.dumps({
                "images": [],
                "categories": [{"id": 1, "name": "player"}],
                "annotations": [],
            }))
            with self.assertRaisesRegex(ValueError, "CVAT images differ"):
                convert_reviewed_cvat(
                    coco, manifest, root / "out.json", confirmed_reviewed=True,
                )


if __name__ == "__main__":
    unittest.main()
