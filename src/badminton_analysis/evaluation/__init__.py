"""Offline evaluation helpers for court-player detection and segmentation."""

from badminton_analysis.evaluation.players import (
    FrameRecord,
    PlayerRecord,
    evaluate_frames,
    load_frames,
)

__all__ = ["FrameRecord", "PlayerRecord", "evaluate_frames", "load_frames"]
