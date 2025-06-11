from scipy import sparse
import numpy as np
import warnings

def row_norms(X, squared=False):
    """
    Compute the L2 norm (or squared norm) of each row in matrix X.
    """
    norms_squared = np.einsum('ij,ij->i', X, X)
    if squared:
        return norms_squared
    else:
        return np.sqrt(norms_squared)

def safe_sparse_dot(a, b, *, dense_output=False):
    """Dot product that handle the sparse matrix case correctly.
    """

    # sparse is always 2D
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a): # so b is 3D+
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # so a is 3D+
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (
        sparse.issparse(a)
        and sparse.issparse(b)
        and dense_output
        and hasattr(ret, "toarray")
    ):
        return ret.toarray()
    return ret

def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.

    Warns if the final cumulative sum does not match the sum (up to the chosen
    tolerance).

    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat.
    axis : int, default=None
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float, default=1e-05
        Relative tolerance, see ``np.allclose``.
    atol : float, default=1e-08
        Absolute tolerance, see ``np.allclose``.

    Returns
    -------
    out : ndarray
        Array with the cumulative sums along the chosen axis.
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.allclose(
        out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
    ):
        warnings.warn(
            (
                "cumsum was found to be unstable: "
                "its last element does not correspond to sum"
            ),
            RuntimeWarning,
        )
    return out