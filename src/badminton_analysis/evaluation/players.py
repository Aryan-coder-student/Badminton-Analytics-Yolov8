"""Ground-truth evaluation independent of tracking, ReID, and YOLO inference."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, slots=True, eq=False)
class PlayerRecord:
    """One player box (XYXY pixels), optionally with a full-frame binary mask."""

    bbox: tuple[float, float, float, float]
    mask: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """Annotated or predicted players from an absolute video frame index."""

    frame_index: int
    players: tuple[PlayerRecord, ...]
    camera_view: str = "unknown"


@dataclass
class EvaluationCounts:
    """Accumulate true/false detections and segmentation quality."""

    frames: int = 0
    reference_players: int = 0
    predicted_players: int = 0
    matched_players: int = 0
    positive_frames: int = 0
    complete_positive_frames: int = 0
    four_player_frames: int = 0
    complete_four_player_frames: int = 0
    box_iou_sum: float = 0.0
    mask_iou_sum: float = 0.0
    matched_mask_pairs: int = 0
    reference_masks: int = 0
    matched_reference_masks_without_prediction: int = 0

    def as_dict(self) -> dict[str, float | int | None]:
        false_positives = self.predicted_players - self.matched_players
        false_negatives = self.reference_players - self.matched_players
        return {
            "frames": self.frames,
            "reference_players": self.reference_players,
            "predicted_players": self.predicted_players,
            "true_positives": self.matched_players,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": _safe_ratio(self.matched_players, self.predicted_players),
            "recall": _safe_ratio(self.matched_players, self.reference_players),
            "f1": _safe_ratio(
                2 * self.matched_players,
                2 * self.matched_players + false_positives + false_negatives,
            ),
            "mean_matched_box_iou": _safe_ratio(self.box_iou_sum, self.matched_players),
            "mean_matched_mask_iou": _safe_ratio(self.mask_iou_sum, self.matched_mask_pairs),
            "reference_masks": self.reference_masks,
            "matched_mask_pairs": self.matched_mask_pairs,
            "matched_reference_masks_without_prediction": (
                self.matched_reference_masks_without_prediction
            ),
            "positive_frames": self.positive_frames,
            "complete_positive_frames": self.complete_positive_frames,
            "all_players_found_rate": _safe_ratio(
                self.complete_positive_frames, self.positive_frames
            ),
            "four_player_frames": self.four_player_frames,
            "complete_four_player_frames": self.complete_four_player_frames,
            "four_player_coverage": _safe_ratio(
                self.complete_four_player_frames, self.four_player_frames
            ),
        }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union for valid XYXY pixel boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = (first[2] - first[0]) * (first[3] - first[1])
    area_b = (second[2] - second[0]) * (second[3] - second[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Pixel IoU; fail explicitly rather than resizing a misaligned mask."""
    if first.shape != second.shape:
        raise ValueError(f"Mask shape mismatch: {first.shape} != {second.shape}")
    first_bool, second_bool = first.astype(bool), second.astype(bool)
    union = np.logical_or(first_bool, second_bool).sum()
    if union == 0:
        raise ValueError("An empty/empty mask pair is invalid for a matched player.")
    return float(np.logical_and(first_bool, second_bool).sum() / union)


def match_players(
    reference: Sequence[PlayerRecord],
    predicted: Sequence[PlayerRecord],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality, one-to-one box-IoU matches with IoU tie breaking."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")
    if not reference or not predicted:
        return []
    reference_size = len(reference)
    predicted_size = len(predicted)
    n = reference_size + predicted_size
    costs = np.zeros((n, n), dtype=np.float64)
    # Bonus dominates any possible summed IoU difference, so valid match count
    # wins before IoU quality. Dummy rows/columns permit unmatched boxes.
    bonus = min(reference_size, predicted_size) + 1
    costs[:reference_size, :predicted_size] = 1.0
    scores: dict[tuple[int, int], float] = {}
    for row, truth in enumerate(reference):
        for col, detection in enumerate(predicted):
            score = box_iou(truth.bbox, detection.bbox)
            if score >= iou_threshold:
                costs[row, col] = -(bonus + score)
                scores[(row, col)] = score
    rows, columns = linear_sum_assignment(costs)
    return [
        (int(row), int(col), scores[(int(row), int(col))])
        for row, col in zip(rows, columns, strict=True)
        if (int(row), int(col)) in scores
    ]


def evaluate_frames(
    reference: Sequence[FrameRecord],
    predicted: Sequence[FrameRecord],
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Evaluate annotated frames only; missing predictions count as zero players."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")
    reference_by_index = _unique_frames(reference)
    predicted_by_index = _unique_frames(predicted)
    if not reference_by_index:
        raise ValueError("Evaluation requires at least one annotated frame")
    unknown_prediction_frames = predicted_by_index.keys() - reference_by_index.keys()
    if unknown_prediction_frames:
        raise ValueError(
            f"Predictions contain unannotated frames: {sorted(unknown_prediction_frames)}"
        )

    overall = EvaluationCounts()
    by_view: dict[str, EvaluationCounts] = defaultdict(EvaluationCounts)
    for frame_index, truth in sorted(reference_by_index.items()):
        detected = predicted_by_index.get(frame_index)
        detections = detected.players if detected else ()
        matches = match_players(truth.players, detections, iou_threshold)
        for counts in (overall, by_view[truth.camera_view]):
            counts.frames += 1
            counts.reference_players += len(truth.players)
            counts.predicted_players += len(detections)
            counts.matched_players += len(matches)
            counts.box_iou_sum += sum(score for _, _, score in matches)
            counts.reference_masks += sum(player.mask is not None for player in truth.players)
            if truth.players:
                counts.positive_frames += 1
                if len(matches) == len(truth.players):
                    counts.complete_positive_frames += 1
            if len(truth.players) == 4:
                counts.four_player_frames += 1
                if len(matches) == 4:
                    counts.complete_four_player_frames += 1
            for row, col, _ in matches:
                truth_mask = truth.players[row].mask
                if truth_mask is None:
                    continue
                pred_mask = detections[col].mask
                if pred_mask is None:
                    counts.matched_reference_masks_without_prediction += 1
                else:
                    counts.matched_mask_pairs += 1
                    counts.mask_iou_sum += mask_iou(truth_mask, pred_mask)
    return {
        "iou_threshold": iou_threshold,
        "overall": overall.as_dict(),
        "by_camera_view": {
            name: counts.as_dict() for name, counts in sorted(by_view.items())
        },
    }


def _unique_frames(frames: Sequence[FrameRecord]) -> dict[int, FrameRecord]:
    records: dict[int, FrameRecord] = {}
    for frame in frames:
        if frame.frame_index < 0:
            raise ValueError("Frame indices must be nonnegative")
        if frame.frame_index in records:
            raise ValueError(f"Duplicate frame index: {frame.frame_index}")
        records[frame.frame_index] = frame
    return records


def load_frames(path: Path) -> list[FrameRecord]:
    """Load JSON player records and optional single-channel PNG masks."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
        raise ValueError("Expected an object containing a 'frames' list")
    records: list[FrameRecord] = []
    for frame in data["frames"]:
        if type(frame.get("frame_index")) is not int:
            raise ValueError("frame_index must be an integer")
        players: list[PlayerRecord] = []
        for player in frame["players"]:
            values = player["bbox"]
            if len(values) != 4 or any(not np.isfinite(float(v)) for v in values):
                raise ValueError("Each bbox must have four finite XYXY coordinates")
            bbox = tuple(float(value) for value in values)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError("Every bbox must have positive area")
            mask = None
            if player.get("mask_path") is not None:
                relative = Path(player["mask_path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("mask_path must be relative to the JSON file directory")
                mask_file = path.parent / relative
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Cannot read mask: {mask_file}")
                if not np.any(mask):
                    raise ValueError(f"Empty player mask: {mask_file}")
                mask = mask > 0
            players.append(PlayerRecord(bbox, mask))
        records.append(FrameRecord(
            frame_index=frame["frame_index"],
            players=tuple(players),
            camera_view=str(frame.get("camera_view", "unknown")),
        ))
    _unique_frames(records)
    return records
