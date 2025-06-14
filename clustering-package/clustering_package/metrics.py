import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Coefficient of all samples.

    The Silhouette Coefficient is calculated using the mean intra-cluster distance (a)
    and the mean nearest-cluster distance (b) for each sample.
    The Silhouette Coefficient for a sample is (b - a) / max(a, b).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A feature array.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    silhouette_avg : float
        The mean Silhouette Coefficient over all samples.
    """
    # Filter out noise points (label -1)
    non_noise_indices = labels != -1
    X_filtered = X[non_noise_indices]
    labels_filtered = labels[non_noise_indices]

    unique_labels = np.unique(labels_filtered)
    if len(unique_labels) < 2 or len(unique_labels) > len(X_filtered) - 1:
        return 0.0

    n_samples_filtered = X_filtered.shape[0]
    a = np.zeros(n_samples_filtered)
    b = np.zeros(n_samples_filtered)

    for i in range(n_samples_filtered):
        current_label = labels_filtered[i]
        current_point = X_filtered[i]

        # Calculate a(i) - mean distance to all other points in the same cluster
        current_cluster_points = X_filtered[labels_filtered == current_label]
        if len(current_cluster_points) > 1:
            a[i] = np.mean([np.linalg.norm(current_point - p) for p in current_cluster_points if not np.array_equal(current_point, p)])
        else:
            a[i] = 0.0 # If a cluster has only one point, a(i) is 0

        # Calculate b(i) - mean distance to all points in the nearest cluster
        other_clusters_labels = [l for l in unique_labels if l != current_label]
        min_b = np.inf
        for other_label in other_clusters_labels:
            other_cluster_points = X_filtered[labels_filtered == other_label]
            if len(other_cluster_points) > 0:
                avg_dist_to_other_cluster = np.mean([np.linalg.norm(current_point - p) for p in other_cluster_points])
                if avg_dist_to_other_cluster < min_b:
                    min_b = avg_dist_to_other_cluster
        b[i] = min_b

    silhouette_scores = np.zeros(n_samples_filtered)
    for i in range(n_samples_filtered):
        if a[i] == 0.0 and b[i] == np.inf: # Case for single point cluster and no other clusters
            silhouette_scores[i] = 0.0
        elif a[i] >= b[i]:
            silhouette_scores[i] = (b[i] - a[i]) / a[i]
        else:
            silhouette_scores[i] = (b[i] - a[i]) / b[i]

    return np.mean(silhouette_scores)


