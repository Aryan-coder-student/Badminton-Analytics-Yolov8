"""BFMD dataset integration tests without model downloads or copyrighted video."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from badminton_analysis.evaluation.bfmd import (
    court_crop_bounds,
    find_bfmd_video,
    nearby_court_hint,
    read_bfmd_pose_prelabels,
    read_court_hints,
    scenes_to_shots,
    select_cluster_frames,
)
from badminton_analysis.evaluation.shot_clustering import (
    ShotRange,
    save_embedding_cache,
)


class BfmdTests(unittest.TestCase):
    def test_video_discovery_and_doubles_kind(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "videos_doubles"
            folder.mkdir()
            (folder / "match-A.mp4").touch()
            (folder / "match-B.mp4").touch()
            video, match, kind = find_bfmd_video(root, "match-A")
            self.assertEqual(video.name, "match-A.mp4")
            self.assertEqual(match, "match-A")
            self.assertEqual(kind, "doubles")
            with self.assertRaisesRegex(ValueError, "Expected one"):
                find_bfmd_video(root, "match")

    def test_transnet_scene_bounds_become_end_exclusive(self) -> None:
        scenes = [
            {"shot_id": 0, "start_frame": 0, "end_frame": 9},
            {"shot_id": 1, "start_frame": 10, "end_frame": 29},
        ]
        self.assertEqual(
            scenes_to_shots(scenes, 30),
            [ShotRange("0", 0, 10), ShotRange("1", 10, 30)],
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            scenes_to_shots([
                {"start_frame": 0, "end_frame": 10},
                {"start_frame": 10, "end_frame": 20},
            ], 30)
        with self.assertRaisesRegex(ValueError, "Invalid"):
            scenes_to_shots([
                {"start_frame": 0, "end_frame": 31},
            ], 30)

    def test_court_hint_roi_only_for_valid_nearby_court(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "match.json"
            path.write_text(json.dumps({"frames": [
                {"frame_idx": 30, "flag": "valid",
                 "corners": [[100, 100], [400, 100], [400, 500], [100, 500]]},
                {"frame_idx": 60, "flag": "no_court", "corners": None},
            ]}))
            hints = read_court_hints(path)
            observed = nearby_court_hint(hints, 35)
            self.assertEqual(observed["frame_idx"], 30)
            bounds = court_crop_bounds(observed, 1280, 720)
            self.assertEqual(bounds, (55, 40, 445, 560))
            self.assertIsNone(court_crop_bounds(hints[1], 1280, 720))
            self.assertIsNone(nearby_court_hint(hints, 1))
            self.assertEqual(read_court_hints(path.parent / "missing.json"), [])

    def test_cluster_candidate_selection_is_unique_and_spread(self) -> None:
        shots = [
            ShotRange(str(n), n * 10, n * 10 + 10)
            for n in range(12)
        ]
        indices = [tuple([n * 10 + 2, n * 10 + 5, n * 10 + 7]) for n in range(12)]
        clusters = np.asarray([n % 3 for n in range(12)])
        selected = select_cluster_frames(shots, clusters, indices, target=15)
        self.assertEqual(len(selected), 15)
        self.assertEqual(len({entry["frame_index"] for entry in selected}), 15)
        self.assertGreaterEqual(len({entry["shot_id"] for entry in selected}), 12)
        self.assertEqual(
            {entry["cluster_id"] for entry in selected}, {0, 1, 2}
        )

    def test_bfmd_runner_with_cached_embeddings_extracts_real_pngs(self) -> None:
        from scripts.prepare_bfmd import run

        with TemporaryDirectory() as directory:
            root = Path(directory) / "BFMD_data"
            videos = root / "videos_doubles"
            videos.mkdir(parents=True)
            video = videos / "Sample-Match.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                30.0, (1280, 720),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV video writer cannot encode mp4v")
            for n in range(30):
                frame = np.full((720, 1280, 3), 100 + n, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            out = Path(directory) / "prepared"
            out.mkdir()
            shots = [ShotRange("1", 0, 15), ShotRange("2", 15, 30)]
            csv_file = out / "shot_ranges.csv"
            csv_file.write_text(
                "shot_id,start_frame,end_frame\n1,0,15\n2,15,30\n"
            )
            sample_indices = [(3, 7, 10), (18, 22, 25)]
            save_embedding_cache(
                out / "shot_embeddings.npz",
                shots,
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                sample_indices,
            )
            args = SimpleNamespace(
                bfmd_root=root, match="Sample-Match", video=None,
                output=out, shot_ranges=csv_file, transnet_weights=None,
                threshold=0.5, clusters=2, frames=4, seed=42,
                max_shots=None, redetect=False, reembed=False,
                allow_video_mismatch=False, camera_view=None,
            )
            with patch(
                "scripts.prepare_bfmd.group_shots",
                return_value=(np.asarray([0, 1]), 0.4),
            ):
                report = run(args)
            self.assertEqual(report["exported_frames"], 4)
            self.assertEqual(report["eligible_shots"], 2)
            self.assertEqual(report["holdout_frames"], 2)
            self.assertEqual(
                len(list((out / "images").glob("*.png"))), 4
            )
            self.assertEqual(
                len(list((out / "cluster_review").glob("*.jpg"))), 2
            )
            template = json.loads(
                (out / "annotations_PRELABELS_NOT_GROUND_TRUTH.json").read_text()
            )
            self.assertTrue(
                all(frame["annotation_status"] == "unlabelled"
                    for frame in template["frames"])
            )
            self.assertTrue(
                all(not frame["players"] for frame in template["frames"])
            )
            self.assertTrue(
                (out / "cluster_camera_views.json").exists()
            )

    def test_pose_box_prelabels_must_still_be_reviewed(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "match.json"
            path.write_text(json.dumps({"videos": {
                "rally_0": {"frames": {
                    "123": [{"bbox": [10, 20, 40, 100]},
                            {"bbox": [30, 5, 30, 60]}],
                    "125": [{"bbox": [5, 15, 60, 70]}],
                }},
            }}))
            prelabels = read_bfmd_pose_prelabels(path, {123})
            self.assertEqual(prelabels, {
                123: [{"bbox": [10.0, 20.0, 40.0, 100.0]}],
            })
            self.assertEqual(
                read_bfmd_pose_prelabels(path.parent / "missing.json", {123}),
                {},
            )


if __name__ == "__main__":
    unittest.main()
