import unittest
import numpy as np
from clustering_package.metrics import silhouette_score

class TestMetrics(unittest.TestCase):

    def test_silhouette_score_perfect_clustering(self):
        X = np.array([
            [0, 0], [0.1, 0.1], [0.2, 0.2],  # Cluster 0
            [10, 10], [10.1, 10.1], [10.2, 10.2] # Cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        score = silhouette_score(X, labels)
        self.assertGreater(score, 0.9) # Should be close to 1 for perfect clustering

    def test_silhouette_score_bad_clustering(self):
        X = np.array([
            [0, 0], [0.1, 0.1], [10, 10],  # Mixed cluster 0
            [0.2, 0.2], [10.1, 10.1], [10.2, 10.2] # Mixed cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        score = silhouette_score(X, labels)
        self.assertLess(score, 0.5) # Should be low for bad clustering

    def test_silhouette_score_single_cluster(self):
        X = np.array([
            [0, 0], [0.1, 0.1], [0.2, 0.2]
        ])
        labels = np.array([0, 0, 0])
        score = silhouette_score(X, labels)
        self.assertEqual(score, 0.0) # Silhouette score is 0 for single cluster

    def test_silhouette_score_noise(self):
        X = np.array([
            [0, 0], [0.1, 0.1], [0.2, 0.2]
        ])
        labels = np.array([-1, -1, -1]) # All noise
        score = silhouette_score(X, labels)
        self.assertEqual(score, 0.0) # Silhouette score is 0 for all noise

    def test_davies_bouldin_index_perfect_clustering(self):
        X = np.array([
            [0, 0], [0.1, 0.1], [0.2, 0.2],  # Cluster 0
            [10, 10], [10.1, 10.1], [10.2, 10.2] # Cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        score = davies_bouldin_index(X, labels)
        self.assertLess(score, 0.5) # Should be low for good clustering

    def test_davies_bouldin_index_bad_clustering(self):
        X = np.array([
            [0, 0], [0.1, 0.1], [10, 10],  # Mixed cluster 0
            [0.2, 0.2], [10.1, 10.1], [10.2, 10.2] # Mixed cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        score = davies_bouldin_index(X, labels)
        self.assertGreater(score, 1.0) # Should be high for bad clustering

    def test_davies_bouldin_index_single_cluster(self):
        X = np.array([
            [0, 0], [0.1, 0.1], [0.2, 0.2]
        ])
        labels = np.array([0, 0, 0])
        score = davies_bouldin_index(X, labels)
        self.assertEqual(score, 0.0) # Davies-Bouldin index is 0 for single cluster

    def test_davies_bouldin_index_noise(self):
        X = np.array([
            [0, 0], [0.1, 0.1], [0.2, 0.2]
        ])
        labels = np.array([-1, -1, -1]) # All noise
        score = davies_bouldin_index(X, labels)
        self.assertEqual(score, 0.0) # Davies-Bouldin index is 0 for all noise

if __name__ == '__main__':
    unittest.main()


