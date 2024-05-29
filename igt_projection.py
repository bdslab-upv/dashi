import datetime

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon, squareform, pdist
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from all_classes import IGTProjection, DataTemporalMap

import numpy as np
from scipy.linalg import eigh

import numpy as np
from scipy.linalg import eigh

from utils import trim_data_temporal_map


def classical_mds_precomputed(dist_matrix, n_components=2):
    """
    Perform Classical Multidimensional Scaling (MDS) using a precomputed distance matrix.
    Parameters:
    - dist_matrix: ndarray, shape (n_samples, n_samples)
        The precomputed distance matrix.
    - n_components: int, optional (default=2)
        The number of dimensions in the reduced space.
    Returns:
    - X_transformed: ndarray, shape (n_samples, n_components)
        The data transformed to the new space.
    """
    # Double centering
    n_samples = dist_matrix.shape[0]
    H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    B = -0.5 * H.dot(dist_matrix ** 2).dot(H)
    # Eigen decomposition
    eigvals, eigvecs = eigh(B)
    # Select the top n_components eigenvalues and corresponding eigenvectors
    idx = np.argsort(eigvals)[::-1][:n_components]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Compute the coordinates using positive-eigenvalue components
    L = np.diag(np.sqrt(eigvals))
    X_transformed = eigvecs.dot(L)

    return X_transformed


def cmdscale(d, k=2, eig=False, add=False, x_ret=False, list_=None):
    if np.any(np.isnan(d)):
        raise ValueError("NA values not allowed in 'd'")

    if not list_:
        if eig:
            print("Warning: eig=TRUE is disregarded when list.=FALSE")
        if x_ret:
            print("Warning: x.ret=TRUE is disregarded when list.=FALSE")

    n = d.shape[0] if hasattr(d, 'shape') else len(d)

    if add:
        if hasattr(d, 'shape'):
            d = d.copy()
        else:
            d = np.array(d)
        x = np.square(d)
        x[np.triu_indices(n)] = 0
        x += x.T
    else:
        if hasattr(d, 'shape'):
            x = np.square(d)
        else:
            x = np.square(np.array(d))

    x_centered = x - np.mean(x, axis=0) - np.mean(x, axis=1)[:, np.newaxis] + np.mean(x)

    if add:
        Z = np.zeros((2 * n, 2 * n))
        i = np.arange(n)
        i2 = n + i
        Z[i2, i] = -1
        Z[i, i2] = -x_centered
        Z[i2, i2] = np.sum(np.linalg.eigh(2 * d)[0])
        e = np.linalg.eigh(Z)[0]
        add_c = np.max(np.real(e))

        x_new = np.zeros((n, n))
        non_diag = np.nonzero(~np.eye(n, dtype=bool))
        x_new[non_diag] = np.square(d[non_diag] + add_c)
        x_centered = x_new - np.mean(x_new, axis=0) - np.mean(x_new, axis=1)[:, np.newaxis] + np.mean(x_new)

    e_vals, e_vecs = eigh(-x_centered / 2)
    idx = np.argsort(e_vals)[::-1][:k]
    e_vals = e_vals[idx]
    e_vecs = e_vecs[:, idx]

    k1 = np.sum(e_vals > 0)
    if k1 < k:
        print("Warning: only {} of the first {} eigenvalues are > 0".format(k1, k))
        e_vecs = e_vecs[:, e_vals > 0]
        e_vals = e_vals[e_vals > 0]

    points = e_vecs * np.sqrt(e_vals)[np.newaxis, :]
    if hasattr(d, 'shape'):
        dimnames = (list(range(n)), None)
    else:
        dimnames = (None, None)

    if list_:
        evalus = np.linalg.eigvalsh(-x_centered / 2)
        result = {
            'points': points,
            'eig': evalus if eig else None,
            'x': x_centered if x_ret else None,
            'ac': add_c if add else 0,
            'GOF': np.sum(e_vals) / np.array([np.sum(np.abs(evalus)), np.sum(np.maximum(evalus, 0))])
        }
        return result
    else:
        return points


def isoMDS(d, y=None, k=2, maxit=50, trace=True, tol=1e-3, p=2):
    if np.any(~np.isfinite(d)) and y is None:
        raise ValueError("An initial configuration must be supplied with NA/Infs in 'd'")

    if y is None:
        y = cmdscale(d, k=k)
    if not isinstance(y, np.ndarray) or y.ndim != 2:
        raise ValueError("'y' must be a 2D matrix")

    n = d.shape[0] if hasattr(d, 'shape') else len(d)

    if not hasattr(d, 'shape') or len(d.shape) == 1:
        d = squareform(d)

    if d.shape[0] != d.shape[1]:
        raise ValueError("Distances must be a square matrix")

    if y.shape != (n, k):
        raise ValueError("Invalid initial configuration")

    if np.any(~np.isfinite(y)):
        raise ValueError("Initial configuration must be complete")

    def stress_func(y_flat, d, n, k, p):
        y = y_flat.reshape((n, k))
        d_hat = pdist(y, metric='minkowski', p=p)
        d_hat = squareform(d_hat)
        np.fill_diagonal(d_hat, 0)
        return np.sum((d - d_hat) ** 2)

    y_flat = y.flatten()
    result = minimize(stress_func, y_flat, args=(d, n, k, p), method='BFGS',
                      options={'maxiter': maxit, 'gtol': tol, 'disp': trace})

    y_final = result.x.reshape((n, k))
    stress = result.fun

    return {'points': y_final, 'stress': stress}


# Example usage:
# d = your_distance_matrix_here
# result = cmdscale(d)
# print(result)


# TODO: comprobar nan y  s
# Comprobar, no deberían salir nan, evitar nans
def js_divergence(p, q):
    m = 0.5 * (p + q)
    result = 0.5 * (np.nansum(p * (np.log2(p / m)), axis=0) + np.nansum(q * (np.log2(q / m)), axis=0))

    return result


def igt_projection_core(data_temporal_map=None, dimensions=3, embedding_type='classicalmds'):
    dates = data_temporal_map.dates
    temporal_map = data_temporal_map.probability_map
    number_of_dates = len(dates)

    # TODO: preguntar a vicent

    dissimilarity_matrix = np.zeros((number_of_dates, number_of_dates))
    for i in range(number_of_dates - 1):
        for j in range(i + 1, number_of_dates):
            dissimilarity_matrix[i, j] = np.sqrt(js_divergence(temporal_map[i, :], temporal_map[j, :]))
            # dissimilarity_matrix[i, j] = np.sqrt(jensenshannon(temporal_map[i, :], temporal_map[j, :]))
            dissimilarity_matrix[j, i] = dissimilarity_matrix[i, j]

    embedding_results = None
    # TODO: to test PCA
    if embedding_type == 'classicalmds':
        # TODO copy from chatgpt
        mds = cmdscale(dissimilarity_matrix)
        # mds = classical_mds_precomputed(dissimilarity_matrix)
        embedding_results = mds
    elif embedding_type == 'nonmetricmds':
        # mds = MDS(n_components=dimensions, dissimilarity='euclidean', metric=False)
        # embedding_results = mds.fit_transform(dissimilarity_matrix)
        embedding_results = isoMDS(dissimilarity_matrix, trace=False, k=dimensions)['points']
    elif embedding_type == 'pca':
        pca = PCA(n_components=dimensions)
        embedding_results = pca.fit_transform(dissimilarity_matrix)

    # stress_value = 1 - embedding_results.stress_ if embedding_type == 'classicalmds' else embedding_results.stress_
    # TODO: check for stress of MDS
    stress_value = 0

    igt_projection = IGTProjection(
        data_temporal_map=data_temporal_map,
        projection=embedding_results,
        embedding_type=embedding_type,
        stress=stress_value
    )

    return igt_projection


def estimate_igt_projection(data_temporal_map, dimensions, start_date=None, end_date=None,
                            embedding_type='classicalmds'):
    if data_temporal_map is None:
        raise ValueError('dataTemporalMap of class DataTemporalMap must be provided')

    if dimensions < 2 or dimensions > len(data_temporal_map.dates):
        raise ValueError('dimensions must be between 2 and length(dataTemporalMap@dates)')

    # TODO: comprobar fechas en rango del data temporal map

    if start_date is not None or end_date is not None:
        if start_date is not None and end_date is not None:
            data_temporal_map = trim_data_temporal_map(data_temporal_map, start_date=start_date, end_date=end_date)
        else:
            if start_date is not None:
                data_temporal_map = trim_data_temporal_map(data_temporal_map, start_date=start_date)
            if end_date is not None:
                data_temporal_map = trim_data_temporal_map(data_temporal_map, end_date=end_date)

    if embedding_type not in ['classicalmds', 'nonmetricmds', 'pca']:  # TODO: PCA, AutoEncoder
        raise ValueError('embeddingType must be one of classicalmds, nonmetricmds or pca')

    value = igt_projection_core(
        data_temporal_map=data_temporal_map,
        dimensions=dimensions,
        embedding_type=embedding_type
    )
    return value
