"""Deterministic coverage of percentile sampling, cache and cluster export."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from badminton_analysis.evaluation.camera_views import load_shot_clusters
from badminton_analysis.evaluation.shot_clustering import (
    ShotRange,
    extract_shot_embeddings,
    interior_sample_indices,
    load_embedding_cache,
    load_shot_ranges,
    save_embedding_cache,
    write_shot_clusters,
)


class FakeEmbedder:
    def encode_batch(self, inputs: list[np.ndarray]) -> np.ndarray:
        return np.asarray([
            [float(frame[0, 0, 0]), float(frame[0, 0, 2])]
            for frame in inputs
        ])


class ShotClusteringTests(unittest.TestCase):
    def test_percentile_sampling_stays_within_shot(self) -> None:
        self.assertEqual(
            interior_sample_indices(ShotRange("1", 100, 201)),
            (125, 150, 175),
        )
        self.assertEqual(interior_sample_indices(ShotRange("2", 11, 12)), (11,))
        self.assertEqual(interior_sample_indices(ShotRange("3", 10, 13)), (10, 11, 12))

    def test_reject_overlapping_boundaries(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shots.csv"
            path.write_text("shot_id,start_frame,end_frame\n1,0,4\n2,3,8\n")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_shot_ranges(path)

    def test_decode_sample_average_and_cache_roundtrip(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            video_path = path / "tiny.avi"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32)
            )
            self.assertTrue(writer.isOpened())
            for index in range(16):
                writer.write(np.full(
                    (32, 32, 3), (index * 10, 20, 200 - index * 10),
                    dtype=np.uint8,
                ))
            writer.release()

            shots = [ShotRange("a", 0, 8), ShotRange("b", 8, 16)]
            embeddings, samples = extract_shot_embeddings(
                video_path, shots, FakeEmbedder()
            )
            self.assertEqual(embeddings.shape, (2, 2))
            self.assertEqual(samples[0], (2, 4, 5))

            cache = path / "shot_embeddings.npz"
            save_embedding_cache(cache, shots, embeddings, samples)
            restored_shots, restored_embeddings, restored_samples = (
                load_embedding_cache(cache)
            )
            self.assertEqual(restored_shots, shots)
            np.testing.assert_allclose(restored_embeddings, embeddings)
            self.assertEqual(restored_samples, samples)

            cluster_csv = path / "shot_clusters.csv"
            write_shot_clusters(
                cluster_csv, shots, np.asarray([0, 1]), samples
            )
            clustered_shots = load_shot_clusters(cluster_csv)
            self.assertEqual(
                [shot["shot_id"] for shot in clustered_shots], ["a", "b"]
            )
            with cluster_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["cluster_id"], "0")
            self.assertEqual(rows[1]["sample50_frame"], "12")

    def test_reject_mismatched_cache_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_embedding_cache(
                    Path(directory) / "bad.npz",
                    [ShotRange("a", 0, 1)],
                    np.ones((2, 3)),
                    [(0,)],
                )

    def test_reject_duplicate_shot_ids(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shots.csv"
            path.write_text(
                "shot_id,start_frame,end_frame\n"
                "18,0,4\n18,4,8\n",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate shot_id"):
                load_shot_ranges(path)


if __name__ == "__main__":
    unittest.main()
