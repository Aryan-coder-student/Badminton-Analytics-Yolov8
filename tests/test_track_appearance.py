import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from badminton_analysis.appearance import (
    AppearanceObservation,
    TrackAppearance,
    TrackAppearanceCollector,
    TrackAppearanceConfig,
)
from badminton_analysis.model.detection import Detection
from badminton_analysis.model.segmentation import Segmentation
from badminton_analysis.model.tracking import TrackedSegmentation
from scratchpad.test_track_appearance import save_descriptor_archive


class FakeEmbedder:
    def __init__(self, descriptor: tuple[float, ...] = (3.0, 4.0)) -> None:
        self.descriptor = np.asarray(descriptor, dtype=np.float32)
        self.batch_sizes: list[int] = []

    def encode_batch(self, inputs: list[object]) -> np.ndarray:
        self.batch_sizes.append(len(inputs))
        return np.stack([self.descriptor for _ in inputs])


def create_tracked_player(
    track_id: int,
    mask: np.ndarray,
    bbox: tuple[float, float, float, float] = (4.0, 4.0, 28.0, 36.0),
) -> TrackedSegmentation:
    detection = Detection(bbox, 0.9, 0, "person")
    return TrackedSegmentation(track_id, Segmentation(detection, mask))


def create_valid_mask() -> np.ndarray:
    mask = np.zeros((40, 40), dtype=bool)
    mask[4:36, 4:28] = True
    return mask


class TrackAppearanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((40, 40, 3), (30, 120, 220), dtype=np.uint8)

    def create_collector(
        self,
        sampling_interval: int = 1,
    ) -> TrackAppearanceCollector:
        return TrackAppearanceCollector(
            embedder=FakeEmbedder(),
            config=TrackAppearanceConfig(
                sampling_interval=sampling_interval,
            ),
        )

    def test_observations_are_grouped_without_merging_track_ids(self) -> None:
        first_player = create_tracked_player(2, create_valid_mask())
        second_player = create_tracked_player(5, create_valid_mask())
        collector = self.create_collector()

        collector.add_frame(100, self.image, [first_player, second_player])
        collector.add_frame(101, self.image, [first_player])
        tracks = collector.finalize()

        self.assertEqual(set(tracks), {2, 5})
        self.assertEqual(tracks[2].observation_count, 2)
        self.assertEqual(tracks[5].observation_count, 1)
        self.assertIsNot(tracks[2], tracks[5])

    def test_observation_preserves_exact_segmentation_and_mask(self) -> None:
        mask = create_valid_mask()
        tracked_player = create_tracked_player(3, mask)
        collector = self.create_collector()

        observation = collector.add_frame(
            200,
            self.image,
            [tracked_player],
        )[0]

        self.assertIs(observation.tracked_player, tracked_player)
        self.assertIs(observation.tracked_player.segmentation.mask, mask)
        self.assertIs(
            observation.tracked_player.detection,
            tracked_player.segmentation.detection,
        )
        self.assertEqual(observation.frame_number, 200)

    def test_sampling_interval_is_applied_per_track(self) -> None:
        tracked_player = create_tracked_player(1, create_valid_mask())
        collector = self.create_collector(sampling_interval=3)

        for frame_number in range(10, 16):
            collector.add_frame(frame_number, self.image, [tracked_player])
        track = collector.finalize()[1]

        self.assertEqual(
            [item.frame_number for item in track.observations],
            [10, 13],
        )

    def test_empty_and_invalid_samples_are_ignored_without_size_threshold(self) -> None:
        empty_mask = np.zeros((40, 40), dtype=bool)
        small_mask = empty_mask.copy()
        small_mask[30, 5] = True
        players = [
            create_tracked_player(1, empty_mask),
            create_tracked_player(2, small_mask),
            create_tracked_player(3, create_valid_mask(), (20, 20, 10, 30)),
        ]
        embedder = FakeEmbedder()
        collector = TrackAppearanceCollector(
            embedder=embedder,
        )

        observations = collector.add_frame(1, self.image, players)

        self.assertEqual([item.track_id for item in observations], [2])
        self.assertEqual(set(collector.finalize()), {2})
        self.assertEqual(embedder.batch_sizes, [1])

    def test_mean_descriptors_are_normalized(self) -> None:
        tracked_player = create_tracked_player(8, create_valid_mask())
        track = TrackAppearance(track_id=8)
        for frame_number, descriptor in enumerate(
            (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        ):
            track.add_observation(
                AppearanceObservation(
                    frame_number,
                    tracked_player,
                    np.ones((4, 4, 3), dtype=np.uint8),
                    descriptor,
                    descriptor,
                )
            )

        track.aggregate()

        expected = np.array([1.0, 1.0]) / np.sqrt(2)
        np.testing.assert_allclose(track.final_dinov2_descriptor, expected)
        np.testing.assert_allclose(track.final_hsv_descriptor, expected)
        self.assertAlmostEqual(
            float(np.linalg.norm(track.final_dinov2_descriptor)),
            1.0,
            places=6,
        )

    def test_empty_appearances_do_not_create_an_archive(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "descriptors.npz"

            with self.assertRaisesRegex(ValueError, "without track appearances"):
                save_descriptor_archive(output_path, {})

            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
