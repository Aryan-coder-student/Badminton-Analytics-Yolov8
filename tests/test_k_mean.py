import unittest

import numpy as np

from badminton_analysis.model.clustering.interface import ClusterModel
from badminton_analysis.model.clustering.inference.k_mean import KMean


class KMeanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = np.array(
            [[0.0, 0.0], [0.1, 0.2], [9.8, 10.0], [10.0, 9.9]]
        )

    def test_implements_cluster_model_interface(self) -> None:
        model = KMean(self.embeddings, n_clusters=2)

        self.assertIsInstance(model, ClusterModel)
        self.assertIs(model.fit(), model)

    def test_predict_assigns_separated_points_to_different_clusters(self) -> None:
        model = KMean(self.embeddings, n_clusters=2).fit()

        labels = model.predict(np.array([[0.05, 0.1], [9.9, 9.9]]))

        self.assertNotEqual(labels[0], labels[1])

    def test_fit_predict_is_deterministic(self) -> None:
        first = KMean(self.embeddings, n_clusters=2).fit_predict(self.embeddings)
        second = KMean(self.embeddings, n_clusters=2).fit_predict(self.embeddings)

        np.testing.assert_array_equal(first, second)

if __name__ == "__main__":
    unittest.main()
