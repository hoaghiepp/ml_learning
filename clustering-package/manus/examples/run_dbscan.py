import numpy as np
from clustering_package.dbscan import DBSCAN
from clustering_package.metrics import silhouette_score
import matplotlib.pyplot as plt

# Generate some sample data
X = np.array([
    [1, 1], [1.1, 1.1], [1.2, 1.2],  # Cluster 1
    [5, 5], [5.1, 5.1], [5.2, 5.2],  # Cluster 2
    [10, 10], [10.1, 10.1], [10.2, 10.2], # Cluster 3
    [20, 20], [20.1, 20.1], [0, 10], [10, 0] # Noise
])

# Initialize and fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan.fit(X)

# Get labels
labels = dbscan.labels_

print("DBSCAN Labels:\n", labels)

# Evaluate performance (excluding noise points for silhouette score)
# Silhouette score requires at least 2 clusters and no noise points for meaningful calculation
# Filter out noise points (-1 label) for silhouette score calculation
filtered_X = X[labels != -1]
filtered_labels = labels[labels != -1]

if len(np.unique(filtered_labels)) > 1:
    silhouette_avg = silhouette_score(filtered_X, filtered_labels)
    print(f"DBSCAN Silhouette Score (excluding noise): {silhouette_avg:.2f}")
else:
    print("Not enough clusters (excluding noise) to calculate Silhouette Score.")

# Visualize the clusters
plt.figure(figsize=(8, 6))
# Plot non-noise points
plt.scatter(X[labels != -1, 0], X[labels != -1, 1], c=labels[labels != -1], cmap=\'viridis\', marker=\'o\', s=50, label=\'Data Points\')
# Plot noise points
plt.scatter(X[labels == -1, 0], X[labels == -1, 1], c=\'gray\', marker=\'x\', s=100, label=\'Noise Points\')

plt.title(\'DBSCAN Clustering\')
plt.xlabel(\'Feature 1\')
plt.ylabel(\'Feature 2\')
plt.legend()
plt.grid(True)
plt.savefig(\'dbscan_clustering.png\')
plt.show()


