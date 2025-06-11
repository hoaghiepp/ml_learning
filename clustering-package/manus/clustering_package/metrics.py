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
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) > len(X) - 1:
        return 0.0

    n_samples = X.shape[0]
    a = np.zeros(n_samples)
    b = np.zeros(n_samples)

    for i in range(n_samples):
        # Calculate a(i) - mean distance to all other points in the same cluster
        current_cluster_points = X[labels == labels[i]]
        if len(current_cluster_points) > 1:
            a[i] = np.mean([np.linalg.norm(X[i] - p) for p in current_cluster_points if not np.array_equal(X[i], p)])
        else:
            a[i] = 0.0 # If a cluster has only one point, a(i) is 0

        # Calculate b(i) - mean distance to all points in the nearest cluster
        other_clusters_labels = [l for l in np.unique(labels) if l != labels[i]]
        min_b = np.inf
        for other_label in other_clusters_labels:
            other_cluster_points = X[labels == other_label]
            if len(other_cluster_points) > 0:
                avg_dist_to_other_cluster = np.mean([np.linalg.norm(X[i] - p) for p in other_cluster_points])
                if avg_dist_to_other_cluster < min_b:
                    min_b = avg_dist_to_other_cluster
        b[i] = min_b

    silhouette_scores = np.zeros(n_samples)
    for i in range(n_samples):
        if a[i] == 0.0 and b[i] == np.inf: # Case for single point cluster and no other clusters
            silhouette_scores[i] = 0.0
        elif a[i] >= b[i]:
            silhouette_scores[i] = (b[i] - a[i]) / a[i]
        else:
            silhouette_scores[i] = (b[i] - a[i]) / b[i]

    return np.mean(silhouette_scores)

def davies_bouldin_index(X, labels):
    """
    Compute the Davies-Bouldin index.

    The Davies-Bouldin index is defined as the average similarity measure of each cluster
    with its most similar cluster, where similarity is the ratio of within-cluster distances
    to between-cluster distances.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A feature array.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    davies_bouldin : float
        The Davies-Bouldin index.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2 or n_clusters > len(X) - 1:
        return 0.0

    # Calculate centroids and within-cluster scatter
    centroids = np.array([np.mean(X[labels == label], axis=0) for label in unique_labels])
    s = np.array([np.mean([np.linalg.norm(p - centroids[i]) for p in X[labels == unique_labels[i]]]) for i in range(n_clusters)])

    # Calculate inter-cluster distance and R_ij
    R = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            if s[i] + s[j] == 0:
                R[i, j] = 0.0
            else:
                R[i, j] = (s[i] + s[j]) / np.linalg.norm(centroids[i] - centroids[j])
            R[j, i] = R[i, j]

    # Calculate D_i and Davies-Bouldin index
    D = np.array([np.max(R[i, :]) for i in range(n_clusters)])
    davies_bouldin = np.mean(D)

    return davies_bouldin


