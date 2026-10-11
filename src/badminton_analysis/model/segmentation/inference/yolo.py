from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from badminton_analysis.model.detection import Detection
from badminton_analysis.model.segmentation.interface import (
    Segmentation,
    SegmentationModel,
)
from badminton_analysis.model.settings import get_settings


@dataclass
class YOLOSegmenter(SegmentationModel):
    """Segment objects using an Ultralytics YOLO segmentation model."""

    model: YOLO = field(
        default_factory=lambda: YOLO(get_settings().segmentation_model_name),
        repr=False,
    )
    confidence_threshold: float = get_settings().segmentation_confidence
    class_ids: tuple[int, ...] | None = None
    device: str | None = get_settings().segmentation_device

    def segment_batch(
        self,
        images: list[np.ndarray],
    ) -> list[list[Segmentation]]:
        results = self.model.predict(
            source=images,
            conf=self.confidence_threshold,
            classes=list(self.class_ids) if self.class_ids is not None else None,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )
        return [self._convert_the_segmentation(result) for result in results]

    def _convert_the_segmentation(self, result: Results) -> list[Segmentation]:
        if result.boxes is None or result.masks is None:
            return []

        boxes = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()
        masks = result.masks.data.cpu().numpy()

        return [
            Segmentation(
                detection=Detection(
                    bbox=(
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=self.model.names[int(class_id)],
                ),
                mask=mask >= 0.5,
            )
            for box, confidence, class_id, mask in zip(
                boxes,
                confidences,
                class_ids,
                masks,
                strict=True,
            )
        ]
