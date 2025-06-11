import numpy as np
from scipy.sparse import issparse

from utils.mathext import row_norms, safe_sparse_dot

def euclidean_distances(X, Y, X_norm_squared=None, Y_norm_squared=None, squared=False):
    if X_norm_squared is None:
        XX = row_norms(X, squared=True)[:, np.newaxis]
    else:
        XX = X_norm_squared[:, np.newaxis]

    if Y_norm_squared is None:
        YY = row_norms(Y, squared=True)[np.newaxis, :]
    else:
        YY = Y_norm_squared[np.newaxis, :]

    distances = -2 * safe_sparse_dot(X, Y.T, dense_output=True)
    distances += XX
    distances += YY

    np.maximum(distances, 0, out=distances)
    if X is Y:
        np.fill_diagonal(distances, 0)

    return distances if squared else np.sqrt(distances, out=distances)