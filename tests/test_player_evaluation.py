"""Deterministic, model-free tests for the first-milestone player benchmark."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from badminton_analysis.evaluation.players import (
    FrameRecord,
    PlayerRecord,
    box_iou,
    evaluate_frames,
    load_frames,
    mask_iou,
    match_players,
)


BOX = (0.0, 0.0, 10.0, 10.0)
AWAY = (20.0, 20.0, 30.0, 30.0)


def frame(
    number: int,
    boxes: list[tuple[float, float, float, float]],
    view: str = "wide",
) -> FrameRecord:
    return FrameRecord(number, tuple(PlayerRecord(bbox) for bbox in boxes), view)


class PlayerEvaluationTests(unittest.TestCase):
    def test_box_iou_rejects_no_overlap_and_measures_exact_match(self) -> None:
        self.assertEqual(box_iou(BOX, BOX), 1.0)
        self.assertEqual(box_iou(BOX, AWAY), 0.0)

    def test_missing_prediction_frame_counts_as_missed_players(self) -> None:
        result = evaluate_frames(
            [frame(10, [BOX]), frame(11, [BOX])],
            [frame(10, [BOX])],
        )
        overall = result["overall"]
        self.assertEqual(overall["true_positives"], 1)
        self.assertEqual(overall["false_negatives"], 1)
        self.assertEqual(overall["recall"], 0.5)
        self.assertEqual(overall["all_players_found_rate"], 0.5)

    def test_false_positives_and_camera_view_breakdown(self) -> None:
        truth = [frame(10, [BOX], "wide"), frame(11, [], "closeup")]
        predictions = [frame(10, [BOX, AWAY]), frame(11, [BOX])]
        result = evaluate_frames(truth, predictions)
        self.assertEqual(result["overall"]["false_positives"], 2)
        self.assertEqual(result["by_camera_view"]["wide"]["recall"], 1.0)
        self.assertEqual(result["by_camera_view"]["closeup"]["false_positives"], 1)
        self.assertIsNone(result["by_camera_view"]["closeup"]["recall"])

    def test_four_player_coverage_only_on_four_player_truth_frames(self) -> None:
        reference = [
            frame(1, [BOX, AWAY, (40, 40, 50, 50), (60, 60, 70, 70)]),
            frame(2, [BOX]),
        ]
        detected = [
            frame(1, [BOX, AWAY, (40, 40, 50, 50)]),
            frame(2, [BOX]),
        ]
        overall = evaluate_frames(reference, detected)["overall"]
        self.assertEqual(overall["four_player_frames"], 1)
        self.assertEqual(overall["complete_four_player_frames"], 0)
        self.assertEqual(overall["four_player_coverage"], 0.0)
        self.assertEqual(overall["all_players_found_rate"], 0.5)

    def test_one_to_one_matching_rejects_duplicate_box(self) -> None:
        matches = match_players(
            [PlayerRecord(BOX)], [PlayerRecord(BOX), PlayerRecord(BOX)]
        )
        self.assertEqual(len(matches), 1)
        overall = evaluate_frames(
            [frame(1, [BOX])], [frame(1, [BOX, BOX])]
        )["overall"]
        self.assertEqual(overall["false_positives"], 1)

    def test_mask_iou_and_unavailable_masks_are_reported(self) -> None:
        gt_mask = np.array([[1, 1], [0, 0]], dtype=bool)
        pred_mask = np.array([[1, 0], [0, 1]], dtype=bool)
        self.assertAlmostEqual(mask_iou(gt_mask, pred_mask), 1 / 3)
        truth = [FrameRecord(1, (PlayerRecord(BOX, gt_mask),))]
        pred = [FrameRecord(1, (PlayerRecord(BOX, pred_mask),))]
        self.assertAlmostEqual(
            evaluate_frames(truth, pred)["overall"]["mean_matched_mask_iou"],
            1 / 3,
        )
        missing = [FrameRecord(1, (PlayerRecord(BOX),))]
        overall = evaluate_frames(truth, missing)["overall"]
        self.assertIsNone(overall["mean_matched_mask_iou"])
        self.assertEqual(overall["matched_reference_masks_without_prediction"], 1)
        with self.assertRaises(ValueError):
            mask_iou(gt_mask, np.ones((3, 3), dtype=bool))

    def test_duplicate_unannotated_frames_and_threshold_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate_frames([frame(1, [BOX]), frame(1, [BOX])], [])
        with self.assertRaisesRegex(ValueError, "unannotated"):
            evaluate_frames([frame(1, [BOX])], [frame(2, [BOX])])
        with self.assertRaises(ValueError):
            evaluate_frames([frame(1, [BOX])], [], iou_threshold=0)

    def test_load_json_and_png_masks(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            cv2.imwrite(
                str(path / "mask.png"),
                np.array([[255, 0], [0, 0]], dtype=np.uint8),
            )
            manifest = path / "frames.json"
            manifest.write_text(json.dumps({"frames": [{
                "frame_index": 12,
                "camera_view": "wide",
                "players": [{"bbox": BOX, "mask_path": "mask.png"}],
            }]}))
            loaded = load_frames(manifest)
            self.assertEqual(loaded[0].frame_index, 12)
            self.assertTrue(loaded[0].players[0].mask[0, 0])
            self.assertFalse(loaded[0].players[0].mask[0, 1])

    def test_reject_invalid_annotations(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"frames": [{
                "frame_index": 1,
                "players": [{"bbox": (1, 1, 0, 3)}],
            }]}))
            with self.assertRaisesRegex(ValueError, "positive area"):
                load_frames(path)


if __name__ == "__main__":
    unittest.main()
