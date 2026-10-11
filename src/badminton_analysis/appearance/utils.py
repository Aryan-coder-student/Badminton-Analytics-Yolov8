"""Small, model-free calculations for track-association evidence."""

import numpy as np


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Return cosine similarity for two non-zero descriptor vectors."""
    first_vector = np.asarray(first, dtype=np.float32).reshape(-1)
    second_vector = np.asarray(second, dtype=np.float32).reshape(-1)
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    if denominator == 0:
        raise ValueError("Cosine similarity requires non-zero descriptors.")
    return float(np.dot(first_vector, second_vector) / denominator)


def calculate_frame_gap(
    predecessor_last_frame: int,
    successor_first_frame: int,
) -> int:
    """Return elapsed frames; tracks in consecutive frames have a gap of one."""
    return successor_first_frame - predecessor_last_frame


def calculate_court_distance(
    predecessor_position: np.ndarray,
    successor_position: np.ndarray,
) -> float:
    """Return Euclidean distance between two court-plane positions in metres."""
    predecessor = np.asarray(predecessor_position, dtype=np.float32)
    successor = np.asarray(successor_position, dtype=np.float32)
    return float(np.linalg.norm(successor - predecessor))


def has_temporal_overlap(
    predecessor_first_frame: int,
    predecessor_last_frame: int,
    successor_first_frame: int,
    successor_last_frame: int,
) -> bool:
    """Return whether two local track lifetimes share at least one frame."""
    return (
        predecessor_first_frame <= successor_last_frame
        and successor_first_frame <= predecessor_last_frame
    )
