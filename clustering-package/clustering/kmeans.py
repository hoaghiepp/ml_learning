import numpy as np
import scipy as sp
from metrics.pairwise import euclidean_distances
from utils.mathext import stable_cumsum


def _init_centroids(
        self,
        X,
        x_squared_norms,
        init,
        random_state,
        sample_weight,
        init_size=None,
        n_centroids=None,
):
    n_samples = X.shape[0]
    n_clusters = self.n_clusters if n_centroids is None else n_centroids

    if init_size is not None and init_size < n_samples:
        init_indices = np.random.randint(0, n_samples, init_size)
        X = X[init_indices]
        x_squared_norms = x_squared_norms[init_indices]
        sample_weight = sample_weight[init_indices]

    if isinstance(init, str) and init == 'k-means++':
        centers, _ = _kmeans_plus_plus(X, n_clusters, random_state=random_state, x_squared_norms=x_squared_norms,
                                       sample_weight=sample_weight)
    else: # isinstance(init, str) and init == 'random':
        seeds = random_state.choice(n_samples, size=n_clusters, replace=False, p=sample_weight / sample_weight.sum())
        centers = X[seeds]

    if sp.issparse(centers):
        centers = centers.toarray()

    return centers


def _kmeans_plus_plus(
        X, n_clusters, x_squared_norms, sample_weight, random_state, n_local_trials=None
):
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.shape)

    if n_local_trials is None:
        n_local_trials = 5 + int(np.log(n_clusters))

    center_id = random_state.choice(n_clusters, p=sample_weight / sample_weight.sum())
    indices = np.full(n_clusters, -1, dtype=int)

    if sp.issparse(X):
        centers[0] = X[[center_id]].to_array()
    else:
        centers[0] = X[center_id]
    indices[0] = center_id

    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
    )
    current_pot = closest_dist_sq @ sample_weight

    for cluster in range(1, n_clusters):
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(
            stable_cumsum(sample_weight * closest_dist_sq), rand_vals
        )
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
        )

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[cluster] = X[[best_candidate]].toarray()
        else:
            centers[cluster] = X[best_candidate]
        indices[cluster] = best_candidate

    return centers, indices

class KMeans:
    def fit(self, X, y=None):
        random_state = self.random_state
        init = self.init

        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            X -= X_mean

        best_inertia, best_labels = None, None

        for i in range(self.n_init):


    def predict(self, X):
        pass

    def fit_predict(self, X, y):
        pass

    def __init__(self, n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0,
                 random_state=None, copy_x=True):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x

    def _check_param_input(self, X, default_n_init=None):
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

        self._tol