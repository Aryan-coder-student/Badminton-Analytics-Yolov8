from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoImageProcessor, CLIPVisionModelWithProjection

# pyrefly: ignore [missing-import]
from badminton_analysis.model.embedding.common import (
    ImageInput,
    load_rgb_image,
    resolve_device,
)
# pyrefly: ignore [missing-import]
from badminton_analysis.model.embedding.interface import EmbeddingModel
# pyrefly: ignore [missing-import]
from badminton_analysis.model.settings import get_settings


def _load_default_processor() -> Any:
    return AutoImageProcessor.from_pretrained(get_settings().clip_model_name)


def _load_default_model() -> nn.Module:
    return CLIPVisionModelWithProjection.from_pretrained(
        get_settings().clip_model_name
    )


@dataclass
class CLIP(EmbeddingModel):
    """Create projected CLIP vision embeddings for images."""

    device: str | torch.device | None = None
    processor: Any = field(default_factory=_load_default_processor, repr=False)
    model: nn.Module = field(default_factory=_load_default_model, repr=False)
    _device: torch.device = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._device = resolve_device(self.device)
        self.model.eval().to(self._device)

    def encode(self, input_sample: ImageInput) -> np.ndarray:
        return self.encode_batch([input_sample])[0]

    def encode_batch(self, inputs: list[ImageInput]) -> np.ndarray:
        images = [load_rgb_image(image) for image in inputs]
        model_inputs = self.processor(images=images, return_tensors="pt")
        model_inputs = {
            name: value.to(self._device) for name, value in model_inputs.items()
        }
        with torch.inference_mode():
            outputs = self.model(**model_inputs)
        return outputs.image_embeds.cpu().numpy()
