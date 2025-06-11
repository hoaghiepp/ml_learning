# Clustering Package

A Python package implementing KMeans and DBSCAN clustering algorithms, along with performance evaluation metrics.

## Installation

To install the package, navigate to the root directory of the project and run:

```bash
pip install .
```

Alternatively, you can install it in editable mode for development:

```bash
pip install -e .
```

## Usage

### KMeans Example

```python
import numpy as np
from clustering_package.kmeans import KMeans
from clustering_package.metrics import silhouette_score

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
```

### DBSCAN Example

```python
import numpy as np
from clustering_package.dbscan import DBSCAN
from clustering_package.metrics import silhouette_score

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
filtered_X = X[labels != -1]
filtered_labels = labels[labels != -1]

if len(np.unique(filtered_labels)) > 1:
    silhouette_avg = silhouette_score(filtered_X, filtered_labels)
    print(f"DBSCAN Silhouette Score (excluding noise): {silhouette_avg:.2f}")
else:
    print("Not enough clusters (excluding noise) to calculate Silhouette Score.")
```

## Development

### Running Tests

To run the unit tests, navigate to the `tests` directory and run:

```bash
python -m unittest discover
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.


