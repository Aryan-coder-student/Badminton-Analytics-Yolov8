import cv2
import numpy as np

from badminton_analysis.appearance.models import AppearanceInput
from badminton_analysis.model.segmentation import Segmentation

COLOR_HISTOGRAM_BINS = (12, 8)
BACKGROUND_FILL = 127


def normalize_descriptor(descriptor: np.ndarray) -> np.ndarray:
    """Return an L2-normalized, one-dimensional descriptor."""
    flattened = np.asarray(descriptor, dtype=np.float32).reshape(-1)
    magnitude = np.linalg.norm(flattened)
    if magnitude == 0:
        raise ValueError("Cannot normalize a zero-magnitude descriptor.")
    return flattened / magnitude


def _extract_crop(
    image: np.ndarray,
    segmentation: Segmentation,
) -> tuple[np.ndarray, np.ndarray] | None:
    if segmentation.mask.shape != image.shape[:2]:
        return None

    coordinates = np.asarray(segmentation.detection.bbox)
    if not np.isfinite(coordinates).all():
        return None

    image_height, image_width = image.shape[:2]
    x_min = max(0, round(coordinates[0]))
    y_min = max(0, round(coordinates[1]))
    x_max = min(image_width, round(coordinates[2]))
    y_max = min(image_height, round(coordinates[3]))
    if x_max <= x_min or y_max <= y_min:
        return None

    crop_mask = segmentation.mask[y_min:y_max, x_min:x_max]
    if not np.any(crop_mask):
        return None
    return image[y_min:y_max, x_min:x_max].copy(), crop_mask


def _create_hsv_descriptor(
    crop: np.ndarray,
    crop_mask: np.ndarray,
) -> np.ndarray:
    player_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [player_hsv],
        [0, 1],
        crop_mask.astype(np.uint8),
        COLOR_HISTOGRAM_BINS,
        [0, 180, 0, 256],
    )
    return normalize_descriptor(histogram)


def extract_appearance_input(
    image: np.ndarray,
    segmentation: Segmentation,
) -> AppearanceInput | None:
    """Create validated DINOv2 and HSV inputs from a player segmentation."""
    extracted_crop = _extract_crop(image, segmentation)
    if extracted_crop is None:
        return None

    crop, crop_mask = extracted_crop
    hsv_descriptor = _create_hsv_descriptor(crop, crop_mask)

    masked_crop = crop.copy()
    masked_crop[~crop_mask] = BACKGROUND_FILL
    masked_crop = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2RGB)
    return AppearanceInput(masked_crop, hsv_descriptor)
