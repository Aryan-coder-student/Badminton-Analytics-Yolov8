from dataclasses import dataclass, field
from typing import Self

import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.neighbors import NearestNeighbors

# pyrefly: ignore [missing-import]
from badminton_analysis.model.clustering.interface import ClusterModel

NOISE_LABEL = -1


@dataclass
class DBScan(ClusterModel):
    """Cluster embeddings with DBSCAN and predict from fitted core samples."""

    epsilon: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"
    leaf_size: int = 30
    n_jobs: int | None = None
    _estimator: SklearnDBSCAN = field(init=False, repr=False)
    _predictor: NearestNeighbors | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _core_labels: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._estimator = SklearnDBSCAN(
            eps=self.epsilon,
            min_samples=self.min_samples,
            metric=self.metric,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
        )

    def fit(self) -> Self:
        self._estimator.fit(self.embeddings)
        self._prepare_predictor()
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self._predictor is None:
            return np.full(embeddings.shape[0], NOISE_LABEL, dtype=int)

        distances, indices = self._predictor.kneighbors(embeddings)
        labels = self._core_labels[indices[:, 0]].copy()
        labels[distances[:, 0] > self.epsilon] = NOISE_LABEL
        return labels

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        labels = self._estimator.fit_predict(embeddings)
        self._prepare_predictor()
        return labels

    def _prepare_predictor(self) -> None:
        core_indices = self._estimator.core_sample_indices_
        self._core_labels = self._estimator.labels_[core_indices]
        self._predictor = NearestNeighbors(
            n_neighbors=1,
            metric=self.metric,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
        ).fit(self._estimator.components_)
