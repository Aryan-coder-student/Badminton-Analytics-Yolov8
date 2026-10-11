from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

# pyrefly: ignore [missing-import]
from badminton_analysis.model.clustering.inference import DBScan, HDBScan, KMean
# pyrefly: ignore [missing-import]
from badminton_analysis.model.clustering.interface import ClusterModel


CLUSTER_STRATEGIES: dict[str, type[ClusterModel]] = {
    "dbscan": DBScan,
    "hdbscan": HDBScan,
    "kmeans": KMean,
}


def create_cluster_strategy(
    tag: str,
    embeddings: np.ndarray,
    **options: Any,
) -> ClusterModel:
    """Create a clustering model from a client-facing strategy tag."""

    normalized_tag = tag.casefold()

    strategy_class = CLUSTER_STRATEGIES.get(normalized_tag)

    if strategy_class is None:
        available = ", ".join(sorted(CLUSTER_STRATEGIES))

        raise ValueError(
            f"Unknown clustering strategy '{tag}'. "
            f"Available strategies: {available}."
        )

    return strategy_class(
        embeddings=embeddings,
        **options,
    )


@dataclass
class Cluster:
    """Client-facing interface for clustering strategies."""

    strategy: str
    embeddings: np.ndarray
    options: dict[str, Any] = field(default_factory=dict)

    model: ClusterModel = field(init=False)

    def __post_init__(self) -> None:
        self.model = create_cluster_strategy(
            self.strategy,
            self.embeddings,
            **self.options,
        )

    def fit(self) -> Self:
        """Fit the active clustering model."""
        self.model.fit()
        return self

    def fit_predict(self) -> np.ndarray:
        """Fit the model and return labels for the configured embeddings."""
        return self.model.fit_predict(self.embeddings)

    def predict(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Assign new embeddings using the fitted clustering model."""
        return self.model.predict(embeddings)

    def set_strategy(
        self,
        strategy: str,
        embeddings: np.ndarray | None = None,
        **options: Any,
    ) -> None:
        """Change the active clustering strategy."""

        self.strategy = strategy

        if embeddings is not None:
            self.embeddings = embeddings

        self.options = options

        self.model = create_cluster_strategy(
            self.strategy,
            self.embeddings,
            **self.options,
        )
