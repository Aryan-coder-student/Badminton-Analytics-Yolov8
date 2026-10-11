from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class EmbeddingModel(ABC):

    @abstractmethod
    def encode(self, input_sample) -> np.ndarray:
        """
        Encode a single sample.

        Returns:
            Embedding of shape (D,)
        """
        pass

    @abstractmethod
    def encode_batch(self, inputs) -> np.ndarray:
        """
        Encode multiple samples in bulk.

        Returns:
            Embeddings of shape (N, D)
        """
        pass
