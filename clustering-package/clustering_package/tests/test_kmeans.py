import unittest
import numpy as np
from clustering_package.model.kmeans import KMeans

class TestKMeans(unittest.TestCase):

    def test_initialization(self):
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.assertEqual(kmeans.n_clusters, 3)
        self.assertEqual(kmeans.random_state, 42)

    def test_fit_predict(self):
        X = np.array([
            [1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5]
        ])
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(X)
        labels = kmeans.predict(X)

        self.assertIsNotNone(kmeans.cluster_centers_)
        self.assertEqual(len(kmeans.cluster_centers_), 2)
        self.assertEqual(len(labels), len(X))
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < 2))

        # Test with a simple, clearly separable dataset
        X_separable = np.array([
            [0, 0], [0.5, 0.5], [1, 0],  # Cluster 0
            [10, 10], [10.5, 10.5], [11, 10] # Cluster 1
        ])
        kmeans_sep = KMeans(n_clusters=2, random_state=0)
        kmeans_sep.fit(X_separable)
        labels_sep = kmeans_sep.predict(X_separable)

        # Check if points from first group are in one cluster and second group in another
        self.assertNotEqual(labels_sep[0], labels_sep[3])
        self.assertEqual(labels_sep[0], labels_sep[1])
        self.assertEqual(labels_sep[3], labels_sep[4])

    def test_predict_unfitted(self):
        kmeans = KMeans(n_clusters=2)
        X = np.array([[1, 1]])
        with self.assertRaises(RuntimeError):
            kmeans.predict(X)

    def test_empty_cluster_handling(self):
        # Test case where a cluster might become empty
        X = np.array([
            [1, 1], [1.1, 1.1], [1.2, 1.2],
            [10, 10], [10.1, 10.1], [10.2, 10.2]
        ])
        kmeans = KMeans(n_clusters=3, random_state=0) # More clusters than natural groups
        kmeans.fit(X)
        labels = kmeans.predict(X)
        self.assertEqual(len(np.unique(labels)), 3 ) # Should still find 2 clusters

if __name__ == '__main__':
    unittest.main()


