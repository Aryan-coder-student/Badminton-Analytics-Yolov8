from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

ImageInput = str | Path | Image.Image | np.ndarray | torch.Tensor


def load_rgb_image(image: ImageInput) -> Image.Image:
    """Convert a supported image input into an RGB PIL image."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened_image:
            return opened_image.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return to_pil_image(image.detach().cpu()).convert("RGB")


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Choose the best available inference device when none is supplied."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
