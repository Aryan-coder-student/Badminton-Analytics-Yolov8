from dataclasses import dataclass, field
from typing import Any

import numpy as np

# pyrefly: ignore [missing-import]
from badminton_analysis.model.embedding.inference import CLIP, DINOv2, ResNet
# pyrefly: ignore [missing-import]
from badminton_analysis.model.embedding.interface import EmbeddingModel


EMBEDDING_STRATEGIES: dict[str, type[EmbeddingModel]] = {
    "clip": CLIP,
    "dinov2": DINOv2,
    "resnet": ResNet,
}


def create_embedding_strategy(
    tag: str,
    **options: Any,
) -> EmbeddingModel:
    """Create an embedding model from a client-facing strategy tag."""

    normalized_tag = tag.casefold()

    strategy_class = EMBEDDING_STRATEGIES.get(normalized_tag)

    if strategy_class is None:
        available = ", ".join(sorted(EMBEDDING_STRATEGIES))

        raise ValueError(
            f"Unknown embedding strategy '{tag}'. "
            f"Available strategies: {available}."
        )

    return strategy_class(**options)


@dataclass
class Embedding:
    """Client-facing interface for embedding strategies."""

    strategy: str
    options: dict[str, Any] = field(default_factory=dict)

    model: EmbeddingModel = field(init=False)

    def __post_init__(self) -> None:
        self.model = create_embedding_strategy(
            self.strategy,
            **self.options,
        )

    def encode(self, input_sample: object) -> np.ndarray:
        """Encode a single input sample."""
        return self.model.encode(input_sample)

    def encode_batch(
        self,
        inputs: list[object],
    ) -> np.ndarray:
        """Encode multiple input samples."""
        return self.model.encode_batch(inputs)

    def set_strategy(
        self,
        strategy: str,
        **options: Any,
    ) -> None:
        """Change the active embedding strategy."""

        self.strategy = strategy
        self.options = options

        self.model = create_embedding_strategy(
            strategy,
            **options,
        )
