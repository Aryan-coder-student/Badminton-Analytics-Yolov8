from dataclasses import dataclass, field
from typing import Self

import numpy as np
from sklearn.cluster import KMeans

# pyrefly: ignore [missing-import]
from badminton_analysis.model.clustering.interface import ClusterModel


@dataclass
class KMean(ClusterModel):
    """Cluster embedding vectors with scikit-learn's K-means algorithm."""

    n_clusters: int = 2
    random_state: int | None = 42
    n_init: int | str = "auto"
    max_iter: int = 300
    tolerance: float = 1e-4
    _estimator: KMeans = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._estimator = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tolerance,
        )

    def fit(self) -> Self:
        self._estimator.fit(self.embeddings)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        predictions=self._estimator.predict(embeddings)
        return predictions

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        predictions=self._estimator.fit_predict(embeddings)
        return predictions
