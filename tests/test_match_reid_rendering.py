import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from badminton_analysis.model.detection import Detection
from badminton_analysis.model.segmentation import Segmentation
from badminton_analysis.model.tracking import TrackedSegmentation
from scratchpad.render_match_reid import (
    FrameIdentityResult,
    annotate_tracked_frame,
    build_ffmpeg_command,
    create_identity_lookup,
    load_frame_results,
    save_frame_results,
)
from scratchpad.tracked_shot import TrackedShotFrame


class MatchReIDRenderingTest(unittest.TestCase):
    def test_final_encoding_preserves_audio_and_uses_h264(self) -> None:
        command = build_ffmpeg_command(
            Path("silent.mp4"),
            Path("source.mp4"),
            Path("output.mp4"),
        )

        self.assertIn("libopenh264", command)
        self.assertIn("1:a?", command)
        self.assertIn("copy", command)
        self.assertEqual(command[-1], "output.mp4")

    def test_saved_identity_results_round_trip(self) -> None:
        records = [FrameIdentityResult(23, 7000, 4, "P2", (1.0, 2.0, 3.0, 4.0))]
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "identities.json"

            save_frame_results(path, records)
            loaded_records = load_frame_results(path)

        self.assertEqual(loaded_records, records)

    def test_identity_lookup_is_scoped_by_shot_and_track(self) -> None:
        records = [
            FrameIdentityResult(18, 1, 2, "P2", (0.0, 0.0, 1.0, 1.0)),
            FrameIdentityResult(23, 2, 2, "P4", (0.0, 0.0, 1.0, 1.0)),
        ]

        identities = create_identity_lookup(records)

        self.assertEqual(identities, {(18, 2): "P2", (23, 2): "P4"})

    def test_annotation_draws_player_mask_and_court_point(self) -> None:
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        mask = np.zeros((60, 80), dtype=bool)
        mask[20:50, 20:40] = True
        detection = Detection((20.0, 20.0, 40.0, 50.0), 0.9, 0, "person")
        tracked_player = TrackedSegmentation(
            3,
            Segmentation(detection, mask),
        )
        tracked_frame = TrackedShotFrame(
            frame_number=100,
            fps=30.0,
            image=image,
            court_grid=np.array([[[60.0, 45.0]]]),
            tracked_players=[tracked_player],
        )

        annotated = annotate_tracked_frame(18, tracked_frame, {(18, 3): "P1"})

        self.assertTrue(np.any(annotated[25:45, 25:35]))
        self.assertTrue(np.any(annotated[43:48, 58:63]))
        self.assertFalse(np.any(image))


if __name__ == "__main__":
    unittest.main()
