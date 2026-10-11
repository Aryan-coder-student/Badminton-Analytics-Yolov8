"""Image embedding strategy implementations."""

from badminton_analysis.model.embedding.inference.clip import CLIP
from badminton_analysis.model.embedding.inference.dinov2 import DINOv2
from badminton_analysis.model.embedding.inference.resnet import ResNet

__all__ = ["CLIP", "DINOv2", "ResNet"]
