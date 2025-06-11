import unittest
import numpy as np
from clustering_package.dbscan import DBSCAN

class TestDBSCAN(unittest.TestCase):

    def test_initialization(self):
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.assertEqual(dbscan.eps, 0.5)
        self.assertEqual(dbscan.min_samples, 5)

    def test_fit_predict(self):
        X = np.array([
            [1, 1], [1.1, 1.1], [1.2, 1.2],  # Cluster 1
            [5, 5], [5.1, 5.1], [5.2, 5.2],  # Cluster 2
            [10, 10], [10.1, 10.1], [10.2, 10.2], # Cluster 3
            [20, 20], [20.1, 20.1] # Noise
        ])
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        dbscan.fit(X)
        labels = dbscan.labels_

        # Check if clusters are formed correctly
        self.assertEqual(len(np.unique(labels)), 4) # 3 clusters + noise (-1)
        self.assertTrue(-1 in labels) # Should have noise points

        # Check specific points
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[3], labels[4])
        self.assertEqual(labels[6], labels[7])
        self.assertEqual(labels[9], -1) # Noise point

        # Test prediction on new data
        X_new = np.array([
            [1.15, 1.15], # Should be in cluster 1
            [5.05, 5.05], # Should be in cluster 2
            [15, 15]      # Should be noise
        ])
        new_labels = dbscan.predict(X_new)
        self.assertEqual(new_labels[0], labels[0])
        self.assertEqual(new_labels[1], labels[3])
        self.assertEqual(new_labels[2], -1)

    def test_predict_unfitted(self):
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        X = np.array([[1, 1]])
        with self.assertRaises(RuntimeError):
            dbscan.predict(X)

    def test_all_noise(self):
        X = np.array([
            [0, 0], [1, 1], [2, 2], [3, 3]
        ])
        dbscan = DBSCAN(eps=0.1, min_samples=3)
        dbscan.fit(X)
        labels = dbscan.labels_
        self.assertTrue(np.all(labels == -1))

    def test_single_cluster(self):
        X = np.array([
            [0, 0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]
        ])
        dbscan = DBSCAN(eps=0.2, min_samples=3)
        dbscan.fit(X)
        labels = dbscan.labels_
        self.assertEqual(len(np.unique(labels)), 1)
        self.assertEqual(labels[0], 0)

if __name__ == '__main__':
    unittest.main()


