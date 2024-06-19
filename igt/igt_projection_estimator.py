import numpy as np
from numpy.linalg import eigh
from scipy.optimize import minimize
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA

from data_temporal_map.data_temporal_map import trim_data_temporal_map
from igt.igt_projection import IGTProjection


def __classical_mds(dist_matrix, n_components=2):
    n = len(dist_matrix)
    H = np.eye(n) - np.ones((n, n)) / n
    B = -H.dot(dist_matrix ** 2).dot(H) / 2
    eigvals, eigvecs = eigh(B)
    idx = np.argsort(eigvals)[::-1][:n_components]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvecs * np.sqrt(eigvals)


def __js_divergence(p, q):
    m = 0.5 * (p + q)
    result = 0.5 * (np.nansum(p * (np.log2(p / m)), axis=0) + np.nansum(q * (np.log2(q / m)), axis=0))

    return result


def __cmdscale(D):
    """
    Classical multidimensional scaling (MDS)
    D   : array [n x n]
        Symmetric distance matrix.

    Returns
    Y   : array [n x p]
        Configuration matrix. Each column represents a dimension.
    """
    n = len(D)

    H = np.eye(n) - np.ones((n, n)) / n
    B = -H.dot(D ** 2).dot(H) / 2

    evals, evecs = np.linalg.eigh(B)

    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    w, = np.where(evals > 0)
    L = np.diag(np.sqrt(evals[w]))
    V = evecs[:, w]
    Y = V.dot(L)

    return Y


def igt_projection_core(data_temporal_map=None, dimensions=3, embedding_type='classicalmds'):
    dates = data_temporal_map.dates
    temporal_map = data_temporal_map.probability_map
    number_of_dates = len(dates)

    dissimilarity_matrix = np.zeros((number_of_dates, number_of_dates))
    for i in range(number_of_dates - 1):
        for j in range(i + 1, number_of_dates):
            dissimilarity_matrix[i, j] = np.sqrt(__js_divergence(temporal_map[i, :], temporal_map[j, :]))
            # dissimilarity_matrix[i, j] = np.sqrt(jensenshannon(temporal_map[i, :], temporal_map[j, :]))
            dissimilarity_matrix[j, i] = dissimilarity_matrix[i, j]

    embedding_results = None
    # TODO: to test PCA
    if embedding_type == 'classicalmds':
        mds = __classical_mds(dissimilarity_matrix)

        embedding_results = mds
    elif embedding_type == 'nonmetricmds':
        # embedding_results, _ = isoMDS(d=dissimilarity_matrix, k=dimensions)
        raise NotImplementedError
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
