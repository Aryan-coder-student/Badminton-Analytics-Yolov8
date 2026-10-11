from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

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


def _get_default_weights() -> ResNet50_Weights:
    return ResNet50_Weights[get_settings().resnet_weights_name]


def _load_default_preprocess() -> Callable[[Image.Image], torch.Tensor]:
    return _get_default_weights().transforms()


def _load_default_model() -> nn.Module:
    model = resnet50(weights=_get_default_weights())
    model.fc = nn.Identity()
    return model


@dataclass
class ResNet(EmbeddingModel):
    """Create image embeddings from ResNet-50 pooled features."""

    device: str | torch.device | None = None
    preprocess: Callable[[Image.Image], torch.Tensor] = field(
        default_factory=_load_default_preprocess,
        repr=False,
    )
    model: nn.Module = field(default_factory=_load_default_model, repr=False)
    _device: torch.device = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._device = resolve_device(self.device)
        self.model.eval().to(self._device)

    def encode(self, input_sample: ImageInput) -> np.ndarray:
        return self.encode_batch([input_sample])[0]

    def encode_batch(self, inputs: list[ImageInput]) -> np.ndarray:
        images = [load_rgb_image(image) for image in inputs]
        batch = torch.stack([self.preprocess(image) for image in images])
        with torch.inference_mode():
            embeddings = self.model(batch.to(self._device))
        return embeddings.cpu().numpy()
