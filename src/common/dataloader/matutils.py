import math

import scipy.linalg
import scipy.sparse
import numpy as np

"""
matutils stolen from gensim.
See https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/matutils.py
"""


def blas(name, ndarray):
    """Helper for getting the appropriate BLAS function, using :func:`scipy.linalg.get_blas_funcs`.

    Parameters
    ----------
    name : str
        Name(s) of BLAS functions, without the type prefix.
    ndarray : numpy.ndarray
        Arrays can be given to determine optimal prefix of BLAS routines.

    Returns
    -------
    object
        BLAS function for the needed operation on the given data type.
    """
    return scipy.linalg.get_blas_funcs((name,), (ndarray,))[0]


blas_nrm2 = blas('nrm2', np.array([], dtype=float))
blas_scal = blas('scal', np.array([], dtype=float))


def unitvec(vec, norm='l2', return_norm=False):
    """Scale a vector to unit length.

    Parameters
    ----------
    vec : {numpy.ndarray, scipy.sparse, list of (int, float)}
        Input vector in any format
    norm : {'l1', 'l2'}, optional
        Metric to normalize in.
    return_norm : bool, optional
        Return the length of vector `vec`, in addition to the normalized vector itself?

    Returns
    -------
    numpy.ndarray, scipy.sparse, list of (int, float)}
        Normalized vector in same format as `vec`.
    float
        Length of `vec` before normalization, if `return_norm` is set.

    Notes
    -----
    Zero-vector will be unchanged.
    """
    if norm not in ('l1', 'l2'):
        raise ValueError("'%s' is not a supported norm. Currently supported norms are 'l1' and 'l2'." % norm)

    if scipy.sparse.issparse(vec):
        vec = vec.tocsr()
        if norm == 'l1':
            veclen = np.sum(np.abs(vec.data))
        else:
            veclen = np.sqrt(np.sum(vec.data ** 2))

        if veclen > 0.0:
            if np.issubdtype(vec.dtype, np.int):
                vec = vec.astype(np.float)
            vec /= veclen
            if return_norm:
                return vec, veclen
            else:
                return vec
        else:
            if return_norm:
                return vec, 1.
            else:
                return vec

    if isinstance(vec, np.ndarray):
        if norm == 'l1':
            veclen = np.sum(np.abs(vec))
        else:
            veclen = blas_nrm2(vec)

        if veclen > 0.0:
            if np.issubdtype(vec.dtype, np.int64):
                vec = vec.astype(np.float)
            if return_norm:
                return blas_scal(1.0 / veclen, vec).astype(vec.dtype), veclen
            else:
                return blas_scal(1.0 / veclen, vec).astype(vec.dtype)
        else:
            if return_norm:
                return vec, 1
            else:
                return vec

    try:
        first = next(iter(vec))  # is there at least one element?
    except StopIteration:
        return vec

    if isinstance(first, (tuple, list)) and len(first) == 2:  # gensim sparse format
        if norm == 'l1':
            length = float(sum(abs(val) for _, val in vec))
        else:
            length = 1.0 * math.sqrt(sum(val ** 2 for _, val in vec))

        assert length > 0.0, "sparse documents must not contain any explicit zero entries"
        if return_norm:
            return ret_normalized_vec(vec, length), length
        else:
            return ret_normalized_vec(vec, length)
    else:
        raise ValueError("unknown input type")


def ret_normalized_vec(vec, length):
    """Normalize a vector in L2 (Euclidean unit norm).
    Parameters
    ----------
    vec : list of (int, number)
        Input vector in BoW format.
    length : float
        Length of vector
    Returns
    -------
    list of (int, number)
        L2-normalized vector in BoW format.
    """
    if length != 1.0:
        return [(termid, val / length) for termid, val in vec]
    else:
        return list(vec)
