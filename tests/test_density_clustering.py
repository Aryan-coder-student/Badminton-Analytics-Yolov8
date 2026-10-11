import unittest

import numpy as np

from badminton_analysis.model.clustering.interface import ClusterModel
from badminton_analysis.model.clustering.inference.dbscan import DBScan
from badminton_analysis.model.clustering.inference.hdbscan import HDBScan


class DensityClusteringTest(unittest.TestCase):
    def setUp(self) -> None:
        first_cluster = np.array(
            [[0.0, 0.0], [0.0, 0.1], [0.1, 0.0], [0.1, 0.1], [0.05, 0.05]]
        )
        second_cluster = first_cluster + 10.0
        self.embeddings = np.vstack((first_cluster, second_cluster))

    def test_dbscan_implements_interface_and_predicts_noise(self) -> None:
        model = DBScan(self.embeddings, epsilon=0.3, min_samples=2)

        self.assertIsInstance(model, ClusterModel)
        self.assertIs(model.fit(), model)
        labels = model.predict(np.array([[0.05, 0.05], [10.05, 10.05], [50, 50]]))

        self.assertNotEqual(labels[0], labels[1])
        self.assertEqual(labels[2], -1)

    def test_dbscan_fit_predict_returns_one_label_per_sample(self) -> None:
        model = DBScan(
            self.embeddings,
            epsilon=0.3,
            min_samples=2,
        )

        labels = model.fit_predict(self.embeddings)
        self.assertEqual(labels.shape, (len(self.embeddings),))
        self.assertEqual(model.predict(np.array([[50, 50]]))[0], -1)

    def test_hdbscan_implements_interface_and_predicts(self) -> None:
        model = HDBScan(self.embeddings, min_cluster_size=3, min_samples=2)

        self.assertIsInstance(model, ClusterModel)
        self.assertIs(model.fit(), model)
        labels = model.predict(np.array([[0.05, 0.05], [10.05, 10.05]]))

        self.assertNotEqual(labels[0], labels[1])

    def test_hdbscan_fit_predict_returns_one_label_per_sample(self) -> None:
        model = HDBScan(
            self.embeddings,
            min_cluster_size=3,
            min_samples=2,
        )

        labels = model.fit_predict(self.embeddings)
        self.assertEqual(labels.shape, (len(self.embeddings),))
        self.assertEqual(model.predict(np.array([[0.05, 0.05]])).shape, (1,))


if __name__ == "__main__":
    unittest.main()
