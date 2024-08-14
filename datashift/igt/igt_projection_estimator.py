import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

from datashift.data_temporal_map.data_temporal_map import trim_data_temporal_map
from datashift.igt.igt_projection import IGTProjection


def __classical_mds(dist_matrix, n_components=2):
    n = len(dist_matrix)
    H = np.eye(n) - np.ones((n, n)) / n
    B = -H.dot(dist_matrix ** 2).dot(H) / 2
    eigvals, eigvecs = eigh(B)
    idx = np.argsort(eigvals)[::-1][:n_components]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvecs * np.sqrt(eigvals)


def __js_divergence(p, q, epsilon=1e-10):
    p = np.asarray(p)
    q = np.asarray(q)

    p = np.where(p < epsilon, epsilon, p)
    q = np.where(q < epsilon, epsilon, q)

    m = 0.5 * (p + q)

    kl_p_m = np.where(p != 0, p * np.log2(p / m), 0)
    kl_q_m = np.where(q != 0, q * np.log2(q / m), 0)

    result = 0.5 * (np.nansum(kl_p_m) + np.nansum(kl_q_m))

    return result


def __cmdscale(d, k=2, eig=False, add=False, x_ret=False):
    # Check for NA values (Not Applicable in numpy, but we can check for NaN)
    if np.isnan(d).any():
        raise ValueError("NA values not allowed in 'd'")

    list_ = eig or add or x_ret

    if not list_:
        if eig:
            print("Warning: eig=TRUE is disregarded when list_=FALSE")
        if x_ret:
            print("Warning: x_ret=TRUE is disregarded when list_=FALSE")

    if not isinstance(d, np.ndarray) or len(d.shape) != 2 or d.shape[0] != d.shape[1]:
        if add:
            d = np.array(d)
        x = np.array(d ** 2, dtype=np.double)
        n = x.shape[0]
        if n != x.shape[1]:
            raise ValueError("distances must be result of 'dist' or a square matrix")
        rn = np.arange(n)
    else:
        n = d.shape[0]
        rn = np.arange(n)
        x = np.zeros((n, n))
        if add:
            d0 = x.copy()
        triu_indices = np.triu_indices_from(x, 1)
        x[triu_indices] = d[triu_indices] ** 2
        x += x.T
        if add:
            d0[triu_indices] = d[triu_indices]
            d = d0 + d0.T

    if not isinstance(n, int) or n > 46340:
        raise ValueError("invalid value of 'n'")

    if k > n - 1 or k < 1:
        raise ValueError("'k' must be in {1, 2, ..  n - 1}")

    # Double centering
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H.dot(x).dot(H)

    if add:
        i2 = n + np.arange(n)
        Z = np.zeros((2 * n, 2 * n))
        Z[np.arange(n), i2] = -1
        Z[i2, np.arange(n)] = -x
        Z[i2, i2] = 2 * d
        e = np.linalg.eigvals(Z)
        add_c = np.max(np.real(e))
        x = np.zeros((n, n), dtype=np.double)
        non_diag = np.triu_indices_from(d, 1)
        x[non_diag] = (d[non_diag] + add_c) ** 2
        x = -0.5 * H.dot(x).dot(H)

    e_vals, e_vecs = eigh(B)
    idx = np.argsort(e_vals)[::-1]
    e_vals = e_vals[idx]
    e_vecs = e_vecs[:, idx]

    ev = e_vals[:k]
    evec = e_vecs[:, :k]
    k1 = np.sum(ev > 0)

    if k1 < k:
        print(f"Warning: only {k1} of the first {k} eigenvalues are > 0")
        evec = evec[:, ev > 0]
        ev = ev[ev > 0]

    points = evec * np.sqrt(ev)

    if list_:
        evalus = e_vals
        return {
            'points': points,
            'eig': evalus if eig else None,
            'x': B if x_ret else None,
            'ac': add_c if add else 0,
            'GOF': np.sum(ev) / np.array([np.sum(np.abs(evalus)), np.sum(np.maximum(evalus, 0))])
        }
    else:
        return points


def igt_projection_core(data_temporal_map=None, dimensions=3, embedding_type='classicalmds'):
    dates = data_temporal_map.dates
    temporal_map = data_temporal_map.probability_map
    number_of_dates = len(dates)

    dissimilarity_matrix = np.zeros((number_of_dates, number_of_dates))
    for i in range(number_of_dates - 1):
        for j in range(i + 1, number_of_dates):
            dissimilarity_matrix[i, j] = np.sqrt(__js_divergence(temporal_map[i, :], temporal_map[j, :]))
            #dissimilarity_matrix[i, j] = jensenshannon(temporal_map[i, :], temporal_map[j, :])
            dissimilarity_matrix[j, i] = dissimilarity_matrix[i, j]

    embedding_results = None
    stress_value = None
    if embedding_type == 'classicalmds':
        mds = __cmdscale(dissimilarity_matrix, k=dimensions)

        embedding_results = mds
    elif embedding_type == 'nonmetricmds':
        nonMDS = MDS(n_components=dimensions,
                     metric=False,
                     random_state=112,
                     dissimilarity='precomputed',
                     normalized_stress='auto',
                     n_init=1)
        embedding_results = nonMDS.fit_transform(dissimilarity_matrix,
                                                 init=(__cmdscale(dissimilarity_matrix, k=dimensions)))
        stress_value = nonMDS.stress_

    elif embedding_type == 'pca':
        scaler = MinMaxScaler()
        scaled_temporal_map = scaler.fit_transform(temporal_map)
        pca = PCA(n_components=dimensions)
        embedding_results = pca.fit_transform(scaled_temporal_map)

    igt_projection = IGTProjection(
        data_temporal_map=data_temporal_map,
        projection=embedding_results,
        embedding_type=embedding_type,
        stress=stress_value
    )

    return igt_projection


def estimate_igt_projection(data_temporal_map, dimensions=2, start_date=None, end_date=None,
                            embedding_type='classicalmds'):
    if data_temporal_map is None:
        raise ValueError('dataTemporalMap of class DataTemporalMap must be provided')

    if dimensions < 2 or dimensions > len(data_temporal_map.dates):
        raise ValueError('dimensions must be between 2 and len(dataTemporalMap.dates)')

    if start_date is not None or end_date is not None:
        if start_date is not None and end_date is not None:
            if start_date and end_date in data_temporal_map.dates:
                data_temporal_map = trim_data_temporal_map(data_temporal_map, start_date=start_date, end_date=end_date)
            else:
                raise ValueError('start_date and end_date must be in the range of dataTemporalMap.dates')
        else:
            if start_date is not None:
                if start_date in data_temporal_map.dates:
                    data_temporal_map = trim_data_temporal_map(data_temporal_map, start_date=start_date)
                else:
                    raise ValueError('start_date must be in the range of dataTemporalMap.dates')
            if end_date is not None:
                if end_date in data_temporal_map.dates:
                    data_temporal_map = trim_data_temporal_map(data_temporal_map, end_date=end_date)
                else:
                    raise ValueError('end_date must be in the range of dataTemporalMap.dates')

    if embedding_type not in ['classicalmds', 'nonmetricmds', 'pca']:  # TODO: PCA, AutoEncoder
        raise ValueError('embeddingType must be one of classicalmds, nonmetricmds or pca')

    value = igt_projection_core(
        data_temporal_map=data_temporal_map,
        dimensions=dimensions,
        embedding_type=embedding_type
    )
    return value
