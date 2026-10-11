import unittest
from unittest.mock import patch

import numpy as np

from badminton_analysis.model.clusters import CLUSTER_STRATEGIES, Cluster
from badminton_analysis.model.clustering.interface import ClusterModel
from badminton_analysis.model.embedding.interface import EmbeddingModel
from badminton_analysis.model.embeddings import EMBEDDING_STRATEGIES, Embedding


class FakeClusterModel(ClusterModel):
    def fit(self) -> None:
        self.was_fitted = True

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return np.full(len(embeddings), 1)

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        return np.full(len(embeddings), 2)


class AlternateClusterModel(FakeClusterModel):
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return np.full(len(embeddings), 3)


class FakeEmbeddingModel(EmbeddingModel):
    def encode(self, input_sample: object) -> np.ndarray:
        del input_sample
        return np.array([1.0, 2.0])

    def encode_batch(self, inputs: list[object]) -> np.ndarray:
        return np.ones((len(inputs), 2))


class AlternateEmbeddingModel(FakeEmbeddingModel):
    def encode(self, input_sample: object) -> np.ndarray:
        del input_sample
        return np.array([3.0, 4.0])


class ConfigurableEmbeddingModel(FakeEmbeddingModel):
    def __init__(self, device: str | None = None) -> None:
        self.device = device


class StrategyContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = np.ones((3, 2))

    def test_cluster_delegates_and_replaces_strategy(self) -> None:
        strategies = {
            "fake": FakeClusterModel,
            "alternate": AlternateClusterModel,
        }
        with patch.dict(CLUSTER_STRATEGIES, strategies):
            context = Cluster("fake", self.embeddings)

            self.assertIs(context.fit(), context)
            self.assertTrue(context.model.was_fitted)
            np.testing.assert_array_equal(
                context.predict(self.embeddings),
                [1, 1, 1],
            )

            context.set_strategy("alternate")
            np.testing.assert_array_equal(
                context.predict(self.embeddings),
                [3, 3, 3],
            )

    def test_cluster_configures_strategy_from_tag(self) -> None:
        with patch.dict(CLUSTER_STRATEGIES, {"fake": FakeClusterModel}):
            context = Cluster("FAKE", self.embeddings)

        np.testing.assert_array_equal(
            context.fit_predict(),
            [2, 2, 2],
        )

    def test_cluster_rejects_unknown_tag(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Available strategies: dbscan, hdbscan, kmeans",
        ):
            Cluster("unknown", self.embeddings)

    def test_embedding_delegates_and_replaces_strategy(self) -> None:
        strategies = {
            "fake": FakeEmbeddingModel,
            "alternate": AlternateEmbeddingModel,
        }
        with patch.dict(EMBEDDING_STRATEGIES, strategies):
            context = Embedding("fake")

            np.testing.assert_array_equal(
                context.encode("frame.jpg"),
                [1.0, 2.0],
            )
            self.assertEqual(
                context.encode_batch(["a.jpg", "b.jpg"]).shape,
                (2, 2),
            )

            context.set_strategy("alternate")
            np.testing.assert_array_equal(
                context.encode("frame.jpg"),
                [3.0, 4.0],
            )

    def test_embedding_configures_strategy_from_tag(self) -> None:
        with patch.dict(EMBEDDING_STRATEGIES, {"fake": FakeEmbeddingModel}):
            context = Embedding("FAKE")

        np.testing.assert_array_equal(context.encode("frame.jpg"), [1.0, 2.0])

    def test_embedding_forwards_options_to_strategy(self) -> None:
        with patch.dict(
            EMBEDDING_STRATEGIES,
            {"configurable": ConfigurableEmbeddingModel},
        ):
            context = Embedding(
                "configurable",
                options={"device": "cpu"},
            )

        self.assertEqual(context.model.device, "cpu")

    def test_embedding_rejects_unknown_tag(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Available strategies: clip, dinov2, resnet",
        ):
            Embedding("unknown")


if __name__ == "__main__":
    unittest.main()
