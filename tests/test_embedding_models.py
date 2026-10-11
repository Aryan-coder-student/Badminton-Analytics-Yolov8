import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn

from badminton_analysis.model.embedding.inference import CLIP, DINOv2, ResNet
from badminton_analysis.model.embedding.interface import EmbeddingModel


class DummyProcessor:
    def __call__(
        self,
        *,
        images: list[Image.Image],
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del return_tensors
        return {"pixel_values": torch.ones(len(images), 3, 2, 2)}


class DummyDINOv2(nn.Module):
    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        batch_size = pixel_values.shape[0]
        hidden_state = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        return SimpleNamespace(last_hidden_state=hidden_state.repeat(batch_size, 1, 1))


class DummyCLIP(nn.Module):
    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        batch_size = pixel_values.shape[0]
        return SimpleNamespace(image_embeds=torch.ones(batch_size, 6))


class DummyResNet(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.ones(images.shape[0], 8)


class EmbeddingModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (8, 8), color="white")

    def test_dinov2_uses_cls_token(self) -> None:
        model = DINOv2(
            device="cpu",
            processor=DummyProcessor(),
            model=DummyDINOv2(),
        )

        self.assertIsInstance(model, EmbeddingModel)
        np.testing.assert_array_equal(model.encode(self.image), np.arange(4))
        self.assertEqual(model.encode_batch([self.image, self.image]).shape, (2, 4))

    def test_clip_returns_projected_image_embeddings(self) -> None:
        model = CLIP(
            device="cpu",
            processor=DummyProcessor(),
            model=DummyCLIP(),
        )

        self.assertIsInstance(model, EmbeddingModel)
        self.assertEqual(model.encode(self.image).shape, (6,))
        self.assertEqual(model.encode_batch([self.image, self.image]).shape, (2, 6))

    def test_resnet_returns_pooled_features(self) -> None:
        preprocess = lambda image: torch.ones(3, image.height, image.width)
        model = ResNet(
            device="cpu",
            preprocess=preprocess,
            model=DummyResNet(),
        )

        self.assertIsInstance(model, EmbeddingModel)
        self.assertEqual(model.encode(self.image).shape, (8,))
        self.assertEqual(model.encode_batch([self.image, self.image]).shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
