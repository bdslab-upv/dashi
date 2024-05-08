import numpy as np
from sklearn.manifold import MDS

from all_classes import IGTProjection


def trim_data_temporal_map(data_temporal_map, start_date=None, end_date=None):
    if start_date is None:
        start_date = data_temporal_map.dates.min()
    if end_date is None:
        end_date = data_temporal_map.dates.max()

    # TODO: duda de fechas
    start_index = data_temporal_map.dates.get_loc(start_date)
    end_index = data_temporal_map.dates.get_loc(end_date)

    temporal_map = data_temporal_map.probability_map[start_index:end_index]
    temporal_counts_map = data_temporal_map.counts_map[start_index:end_index]

    data_temporal_map.probability_map = temporal_map
    data_temporal_map.counts_map = temporal_counts_map

    return data_temporal_map


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
            dissimilarity_matrix[j, i] = dissimilarity_matrix[i, j]

    embedding_results = None
    # TODO: PCA,
    if embedding_type == 'classicalmds':
        mds = MDS(n_components=dimensions, dissimilarity='precomputed')
        # Que pasa con los NA
        embedding_results = mds.fit_transform(dissimilarity_matrix)
    elif embedding_type == 'nonmetricmds':
        embedding_results = MDS(n_components=dimensions, dissimilarity='precomputed', metric=False).fit_transform(dissimilarity_matrix)


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

    if embedding_type not in ['classicalmds', 'nonmetricmds']: #TODO: PCA, AutoEncoder
        raise ValueError('embeddingType must be one of classicalmds or nonmetricmds')

    value = igt_projection_core(data_temporal_map=data_temporal_map, dimensions=dimensions,
                                embedding_type=embedding_type)
    return value
