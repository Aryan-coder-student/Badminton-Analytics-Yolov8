"""Tests for reusing existing visual shot clusters as camera-view prelabels."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from badminton_analysis.evaluation.camera_views import (
    apply_camera_views,
    camera_view_template,
    load_camera_view_map,
    load_shot_clusters,
    load_shot_view_overrides,
)


class CameraViewClusterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.csv = Path(self.temp.name) / "shots.csv"
        self.csv.write_text(
            "shot_id,cluster_id,start_frame,end_frame\n"
            "18,2,100,200\n"
            "23,5,200,300\n",
            encoding="utf-8",
        )
        self.shots = load_shot_clusters(self.csv)

    def test_per_shot_clusters_use_inclusive_start_exclusive_end(self) -> None:
        annotations = {
            "video": "data/sample2.mp4",
            "frames": [
                {"frame_index": 100, "players": [{"bbox": [1, 2, 3, 4]}]},
                {"frame_index": 199, "players": []},
                {"frame_index": 200, "players": []},
            ],
        }
        result = apply_camera_views(
            annotations, self.shots, {"2": "wide", "5": "closeup"}
        )
        self.assertEqual(
            [record["camera_view"] for record in result["frames"]],
            ["wide", "wide", "closeup"],
        )
        self.assertEqual(
            [record["shot_id"] for record in result["frames"]],
            ["18", "18", "23"],
        )
        self.assertEqual(result["frames"][0]["visual_cluster_id"], "2")
        self.assertEqual(annotations["frames"][0].get("camera_view"), None)
        self.assertEqual(annotations["frames"][0]["players"][0]["bbox"], [1, 2, 3, 4])

    def test_template_never_guesses_semantic_camera_views(self) -> None:
        self.assertEqual(camera_view_template(self.shots), {
            "2": "needs_review", "5": "needs_review"
        })

    def test_mapping_requires_explicit_review_or_needs_review(self) -> None:
        path = Path(self.temp.name) / "views.json"
        path.write_text(json.dumps({"2": "wide", "5": "needs_review"}))
        mapping = load_camera_view_map(path, {"2", "5"})
        self.assertEqual(mapping["5"], "needs_review")
        path.write_text(json.dumps({"2": "wide"}))
        with self.assertRaisesRegex(ValueError, "Missing camera view"):
            load_camera_view_map(path, {"2", "5"})
        path.write_text(json.dumps({"2": "MAIN_COURT", "5": "closeup"}))
        with self.assertRaisesRegex(ValueError, "Invalid camera view"):
            load_camera_view_map(path, {"2", "5"})

    def test_manual_view_conflict_requires_review(self) -> None:
        annotations = {"frames": [
            {"frame_index": 123, "players": [], "camera_view": "closeup"}
        ]}
        with self.assertRaisesRegex(ValueError, "already has camera_view"):
            apply_camera_views(annotations, self.shots, {"2": "wide"})

    def test_missing_shot_coverage_fails_instead_of_guessing(self) -> None:
        with self.assertRaisesRegex(ValueError, "no unique shot-cluster"):
            apply_camera_views(
                {"frames": [{"frame_index": 300, "players": []}]},
                self.shots, {"5": "closeup"},
            )

    def test_manual_shot_override_preserves_other_cluster_labels(self) -> None:
        annotations = {"frames": [
            {"frame_index": 123, "players": []},
            {"frame_index": 201, "players": []},
        ]}
        output = apply_camera_views(
            annotations,
            self.shots,
            {"2": "wide", "5": "closeup"},
            shot_overrides={"18": "side_angle"},
        )
        self.assertEqual(
            [frame["camera_view"] for frame in output["frames"]],
            ["side_angle", "closeup"],
        )
        self.assertEqual(
            [frame["camera_view_source"] for frame in output["frames"]],
            ["shot_override", "reviewed_cluster"],
        )

    def test_override_json_rejects_unknown_shot_and_invalid_view(self) -> None:
        path = Path(self.temp.name) / "overrides.json"
        path.write_text(json.dumps({"unknown": "wide"}))
        with self.assertRaisesRegex(ValueError, "Unknown shot_id"):
            load_shot_view_overrides(path, {"18", "23"})
        path.write_text(json.dumps({"18": "FOO"}))
        with self.assertRaisesRegex(ValueError, "Invalid camera view"):
            load_shot_view_overrides(path, {"18", "23"})
        path.write_text(json.dumps({"18": "partial_court"}))
        self.assertEqual(
            load_shot_view_overrides(path, {"18", "23"}),
            {"18": "partial_court"},
        )

    def test_overlapping_shots_rejected(self) -> None:
        self.csv.write_text(
            "shot_id,cluster_id,start_frame,end_frame\n"
            "18,2,100,210\n"
            "23,5,200,300\n",
        )
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            load_shot_clusters(self.csv)

    def test_duplicate_shots_rejected(self) -> None:
        self.csv.write_text(
            "shot_id,cluster_id,start_frame,end_frame\n"
            "18,2,100,150\n"
            "18,5,200,300\n",
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_shot_clusters(self.csv)


if __name__ == "__main__":
    unittest.main()
