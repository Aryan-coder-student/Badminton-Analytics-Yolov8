from dataclasses import dataclass, field
from typing import Self

import numpy as np
from hdbscan import HDBSCAN as HDBSCANEstimator
from hdbscan.prediction import approximate_predict

# pyrefly: ignore [missing-import]
from badminton_analysis.model.clustering.interface import ClusterModel


@dataclass
class HDBScan(ClusterModel):
    """Cluster embeddings with HDBSCAN and support approximate prediction."""

    min_cluster_size: int = 5
    min_samples: int | None = None
    metric: str = "euclidean"
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "eom"
    allow_single_cluster: bool = False
    _estimator: HDBSCANEstimator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._estimator = HDBSCANEstimator(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
            prediction_data=True,
        )

    def fit(self) -> Self:
        self._estimator.fit(self.embeddings)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        labels, _ = approximate_predict(self._estimator, embeddings)
        return labels

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self._estimator.fit_predict(embeddings)
