import numpy as np
from clustering_package.kmeans import KMeans
from clustering_package.metrics import silhouette_score
import matplotlib.pyplot as plt

# Generate some sample data
X = np.array([
    [1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5],
    [8, 8], [9, 8.5], [8.5, 9], [9.5, 9.5]
])

# Initialize and fit KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Predict clusters
labels = kmeans.predict(X)

print("KMeans Cluster Centers:\n", kmeans.cluster_centers_)
print("KMeans Labels:\n", labels)

# Evaluate performance
silhouette_avg = silhouette_score(X, labels)
print(f"KMeans Silhouette Score: {silhouette_avg:.2f}")

# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', s=50, label='Data Points')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Cluster Centers')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.savefig('kmeans_clustering.png')
plt.show()


