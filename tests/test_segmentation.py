import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from badminton_analysis.model.detection import Detection
from badminton_analysis.model.segmentation import Segmentation, SegmentationModel
from badminton_analysis.model.segmentation.inference import YOLOSegmenter
from badminton_analysis.model.segmentations import SEGMENTATION_STRATEGIES, Segmenter
from scratchpad.tracked_shot import (
    PERSON_CLASS_ID,
    SEGMENTATION_CONFIDENCE,
    create_player_segmenter,
)


class FakeYOLO:
    names = {0: "person"}

    def __init__(self, results: list[SimpleNamespace]) -> None:
        self.results = results
        self.predict_arguments: dict[str, object] = {}

    def predict(self, **arguments: object) -> list[SimpleNamespace]:
        self.predict_arguments = arguments
        return self.results


def create_yolo_result() -> SimpleNamespace:
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=torch.tensor([[1.0, 2.0, 5.0, 6.0]]),
            conf=torch.tensor([0.9]),
            cls=torch.tensor([0.0]),
        ),
        masks=SimpleNamespace(
            data=torch.tensor(
                [[[0.0, 1.0], [1.0, 0.0]]],
            )
        ),
    )


class FakeSegmentationModel(SegmentationModel):
    def __init__(self, class_name: str = "person") -> None:
        self.class_name = class_name

    def segment_batch(
        self,
        images: list[np.ndarray],
    ) -> list[list[Segmentation]]:
        detection = Detection((1.0, 2.0, 5.0, 6.0), 0.9, 0, self.class_name)
        segmentation = Segmentation(detection, np.ones((8, 8), dtype=bool))
        return [[segmentation] for _ in images]


class AlternateSegmentationModel(FakeSegmentationModel):
    def __init__(self) -> None:
        super().__init__("sports ball")


class YOLOSegmentationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((8, 8, 3), dtype=np.uint8)

    def test_segment_converts_yolo_mask_and_detection(self) -> None:
        model = FakeYOLO([create_yolo_result()])
        segmenter = YOLOSegmenter(model=model, confidence_threshold=0.5)

        segmentation = segmenter.segment(self.image)[0]

        self.assertIsInstance(segmenter, SegmentationModel)
        self.assertEqual(segmentation.detection.class_name, "person")
        self.assertEqual(segmentation.detection.bbox, (1.0, 2.0, 5.0, 6.0))
        np.testing.assert_array_equal(
            segmentation.mask,
            [[False, True], [True, False]],
        )
        self.assertEqual(model.predict_arguments["conf"], 0.5)
        self.assertTrue(model.predict_arguments["retina_masks"])
        self.assertIsNone(model.predict_arguments["classes"])

    def test_segmenter_passes_class_filter_and_cpu_device_to_yolo(self) -> None:
        model = FakeYOLO([create_yolo_result()])
        segmenter = YOLOSegmenter(
            model=model,
            class_ids=(0,),
            device="cpu",
        )

        segmenter.segment(self.image)

        self.assertEqual(model.predict_arguments["classes"], [0])
        self.assertEqual(model.predict_arguments["device"], "cpu")

    def test_player_segmenter_filters_before_mask_upscaling(self) -> None:
        with patch("scratchpad.tracked_shot.Segmenter") as segmenter_factory:
            create_player_segmenter()

        segmenter_factory.assert_called_once_with(
            "yolo",
            options={
                "confidence_threshold": SEGMENTATION_CONFIDENCE,
                "class_ids": (PERSON_CLASS_ID,),
            },
        )

    def test_segment_batch_returns_results_per_image(self) -> None:
        with_mask = create_yolo_result()
        without_mask = SimpleNamespace(boxes=None, masks=None)
        segmenter = YOLOSegmenter(model=FakeYOLO([with_mask, without_mask]))

        results = segmenter.segment_batch([self.image, self.image])

        self.assertEqual(len(results[0]), 1)
        self.assertEqual(results[1], [])

    def test_segmenter_configures_and_replaces_strategy(self) -> None:
        strategies = {
            "fake": FakeSegmentationModel,
            "alternate": AlternateSegmentationModel,
        }
        with patch.dict(SEGMENTATION_STRATEGIES, strategies):
            segmenter = Segmenter("FAKE", options={"class_name": "player"})
            self.assertEqual(
                segmenter.segment(self.image)[0].detection.class_name,
                "player",
            )

            segmenter.set_strategy("alternate")

        self.assertEqual(
            segmenter.segment(self.image)[0].detection.class_name,
            "sports ball",
        )


if __name__ == "__main__":
    unittest.main()
