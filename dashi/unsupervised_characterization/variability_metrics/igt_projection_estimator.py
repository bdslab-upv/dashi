# Copyright 2024 Biomedical Data Science Lab, Universitat Politècnica de València (Spain)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Functions for Information Geometric Temporal creation
"""

from datetime import datetime
from typing import Optional, Dict, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

from dashi.unsupervised_characterization.data_temporal_map.data_temporal_map import (trim_data_temporal_map,
                                                                                     DataTemporalMap,
                                                                                     MultiVariateDataTemporalMap)
from dashi.unsupervised_characterization.utils import _js_divergence, _cmdscale
from dashi.unsupervised_characterization.variability_metrics.igt_projection import IGTProjection

__all__ = ['estimate_igt_projection']


def _igt_projection_core(data_temporal_map=None, dimensions=3, embedding_type='classicalmds'):
    """
    Computes the core Information Geometric Temporal (IGT) projection for a given DataTemporalMap or
    MultiVariateDataTemporalMap.
    """
    temporal_map = data_temporal_map.probability_map
    nan_rows = np.all(np.isnan(temporal_map), axis=1)
    temporal_map = temporal_map[~nan_rows]
    dates = data_temporal_map.dates
    dates = dates[~nan_rows]
    number_of_dates = len(dates)

    dissimilarity_matrix = np.zeros((number_of_dates, number_of_dates))
    for i in range(number_of_dates - 1):
        for j in range(i + 1, number_of_dates):
            dissimilarity_matrix[i, j] = np.sqrt(_js_divergence(temporal_map[i, :], temporal_map[j, :]))
            dissimilarity_matrix[j, i] = dissimilarity_matrix[i, j]

    # Check if the dissimilarity matrix is all zeros
    if np.all(dissimilarity_matrix == 0):
        raise ValueError("The dissimilarity matrix is all zeros. Cannot compute IGT projection.")

    embedding_results = None
    stress_value = None
    if embedding_type == 'classicalmds':
        embedding_results = _cmdscale(dissimilarity_matrix, k=dimensions)

    elif embedding_type == 'nonmetricmds':
        nonMDS = MDS(n_components=dimensions,
                     metric=False,
                     random_state=112,
                     dissimilarity='precomputed',
                     normalized_stress='auto',
                     n_init=1)
        embedding_results = nonMDS.fit_transform(dissimilarity_matrix,
                                                 init=(_cmdscale(dissimilarity_matrix, k=dimensions)))
        stress_value = nonMDS.stress_

    elif embedding_type == 'pca':
        scaler = MinMaxScaler()
        scaled_temporal_map = scaler.fit_transform(temporal_map)
        pca = PCA(n_components=dimensions)
        embedding_results = pca.fit_transform(scaled_temporal_map)

    projection = np.zeros((len(nan_rows), dimensions))
    projection[~nan_rows] = embedding_results
    igt_projection = IGTProjection(
        data_temporal_map=data_temporal_map,
        projection=projection,
        embedding_type=embedding_type,
        stress=stress_value
    )

    return igt_projection


def _align_and_combine_conditional_temporal_maps(
        conditional_maps: Dict[str, Union[DataTemporalMap, MultiVariateDataTemporalMap]]
) -> DataTemporalMap:
    probability_maps_list: list = []
    periods = set()
    variable_names = set()
    dates_indexes = []
    all_multivariate = all(isinstance(conditional_map, MultiVariateDataTemporalMap)
                           for conditional_map in conditional_maps.values())

    for conditional_map in conditional_maps.values():
        probability_maps_list.append(conditional_map.probability_map)
        dates_indexes.append(pd.DatetimeIndex(pd.to_datetime(conditional_map.dates)))
        periods.add(conditional_map.period)
        variable_names.add(conditional_map.variable_name)

    if len(periods) > 1:
        raise ValueError('All conditional temporal maps must have the same period.')

    dates = pd.DatetimeIndex(np.unique(np.concatenate([dates_index.values for dates_index in dates_indexes])))
    aligned_probability_maps = []
    for probability_map, dates_index in zip(probability_maps_list, dates_indexes):
        aligned_probability_maps.append(pd.DataFrame(probability_map, index=dates_index).reindex(dates).values)

    concatenated_matrix = np.concatenate(aligned_probability_maps, axis=1)
    row_sums = np.nansum(concatenated_matrix, axis=1, keepdims=True)
    normalized_matrix = np.full_like(concatenated_matrix, np.nan, dtype=float)

    valid_mask = np.isfinite(row_sums) & (row_sums > 0)
    np.divide(concatenated_matrix, row_sums, out=normalized_matrix, where=valid_mask)

    variable_name = 'Conditional DTM'
    if not all_multivariate and len(variable_names) == 1:
        variable_name = f'Conditional {next(iter(variable_names))}'

    return DataTemporalMap(
        probability_map=normalized_matrix,
        counts_map=None,
        dates=dates,
        support=None,
        variable_name=variable_name,
        variable_type='float64',
        period=next(iter(periods))
    )


def _resolve_temporal_map_for_igt(
        data_temporal_map: Union[DataTemporalMap, MultiVariateDataTemporalMap, Dict],
        variable_name: Optional[str] = None
) -> Union[DataTemporalMap, MultiVariateDataTemporalMap]:
    if isinstance(data_temporal_map, (DataTemporalMap, MultiVariateDataTemporalMap)):
        if variable_name is not None and variable_name != data_temporal_map.variable_name:
            raise ValueError('variable_name does not match the provided DataTemporalMap variable_name.')
        return data_temporal_map

    if not isinstance(data_temporal_map, dict):
        raise TypeError('data_temporal_map must be a DataTemporalMap, MultiVariateDataTemporalMap, or dictionary.')

    if len(data_temporal_map) == 0:
        raise ValueError('data_temporal_map dictionary must contain at least one map.')

    if variable_name is not None and variable_name in data_temporal_map:
        selected_map = data_temporal_map[variable_name]
        if not isinstance(selected_map, (DataTemporalMap, MultiVariateDataTemporalMap)):
            raise TypeError('The selected variable_name entry must be a temporal map object.')
        return selected_map

    values = list(data_temporal_map.values())

    if all(isinstance(value, MultiVariateDataTemporalMap) for value in values):
        if variable_name is not None:
            raise ValueError('variable_name is only valid for dictionaries of univariate DataTemporalMap objects.')
        return _align_and_combine_conditional_temporal_maps(data_temporal_map)

    if all(isinstance(value, DataTemporalMap) for value in values):
        variable_names = {value.variable_name for value in values}
        if variable_name is not None:
            if variable_name not in variable_names:
                raise ValueError('variable_name was not found as a dictionary key or DataTemporalMap variable_name.')
            conditional_maps = {
                label: conditional_map
                for label, conditional_map in data_temporal_map.items()
                if conditional_map.variable_name == variable_name
            }
            if len(conditional_maps) != len(data_temporal_map):
                raise ValueError('All conditional DataTemporalMap objects must match variable_name.')
            return _align_and_combine_conditional_temporal_maps(conditional_maps)

        if len(variable_names) == 1:
            return _align_and_combine_conditional_temporal_maps(data_temporal_map)

        raise ValueError('variable_name must be provided when data_temporal_map is a dictionary of univariate maps.')

    if all(isinstance(value, dict) for value in values):
        if variable_name is None:
            raise ValueError('variable_name must be provided for conditional maps with multiple variables per label.')

        conditional_maps = dict()
        for label, maps_by_variable in data_temporal_map.items():
            if variable_name not in maps_by_variable:
                raise ValueError(f'Variable {variable_name} not found for label {label}.')
            selected_map = maps_by_variable[variable_name]
            if not isinstance(selected_map, DataTemporalMap):
                raise TypeError('Selected conditional univariate maps must be DataTemporalMap objects.')
            conditional_maps[label] = selected_map

        return _align_and_combine_conditional_temporal_maps(conditional_maps)

    raise TypeError('data_temporal_map dictionary values must be all temporal maps or all dictionaries.')


def estimate_igt_projection(data_temporal_map: Union[DataTemporalMap, MultiVariateDataTemporalMap,
                            Dict[str, MultiVariateDataTemporalMap], Dict[str, DataTemporalMap], Dict],
                            dimensions: int = 2,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            embedding_type: str = 'classicalmds',
                            variable_name: Optional[str] = None
                            ) -> IGTProjection:
    """
    Estimates the Information Geometric Temporal (IGT) projection of a temporal data map, either a
    `DataTemporalMap`, `MultiVariateDataTemporalMap`, a dictionary containing `{label: MultiVariateDataTemporalMap}`,
    a dictionary containing `{variable_name: DataTemporalMap}`, or a conditional univariate dictionary containing
    `{label: DataTemporalMap}` or `{label: {variable_name: DataTemporalMap}}`.

    The IGT projection is a technique to visualize the temporal relationships between data batches
    by projecting the data into a lower-dimensional space (e.g., 2D or 3D), with time batches represented
    as points. The distance between points reflects the probabilistic distance between the data distributions
    of those time batches.

    Parameters
    ----------
    data_temporal_map : Union[DataTemporalMap, MultiVariateDataTemporalMap, Dict]
        The data temporal map to project. This can either be a `DataTemporalMap` object
        (result of estimate_univariate_data_temporal_map), a `MultiVariateDataTemporalMap` object
        (result of estimate_multivariate_data_temporal_map), a dictionary of `DataTemporalMap` objects
        (result of estimate_univariate_data_temporal_map with multiple variables), a dictionary of
        `MultiVariateDataTemporalMap` objects where the keys are labels (result of
        estimate_conditional_data_temporal_map), or a dictionary of conditional univariate maps (result of
        estimate_conditional_univariate_data_temporal_map).

    dimensions : int, optional
        The number of dimensions to use for the projection (2 or 3). Defaults to 2.

    start_date : Optional[datetime], optional
        The starting date for the temporal plot. If None, it is not constrained. Default is None.

    end_date : Optional[datetime], optional
        The ending date for the temporal plot. If None, it is not constrained. Default is None.

    embedding_type : str, optional
        The type of embedding technique to use for dimensionality reduction. Choices are
        'classicalmds' (Classical Multidimensional Scaling), 'pca' (Principal Component Analysis)
        and 'nonmetricmds' (Non Metric Multidimensional Scaling). Defaults to 'classicalmds'.

    variable_name : Optional[str], optional
        Variable to select when `data_temporal_map` is a dictionary returned by
        estimate_univariate_data_temporal_map or estimate_conditional_univariate_data_temporal_map functions.

    Returns
    -------
    IGTProjection
        The estimated IGT projection.
    """
    if data_temporal_map is None:
        raise ValueError('dataTemporalMap must be provided')

    data_temporal_map = _resolve_temporal_map_for_igt(
        data_temporal_map=data_temporal_map,
        variable_name=variable_name
    )

    if dimensions < 2 or dimensions > len(data_temporal_map.dates):
        raise ValueError('dimensions must be between 2 and len(dataTemporalMap.dates)')

    if start_date is not None or end_date is not None:
        if start_date is not None and end_date is not None:
            if start_date in data_temporal_map.dates and end_date in data_temporal_map.dates:
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

    if embedding_type not in ['classicalmds', 'nonmetricmds', 'pca']:
        raise ValueError('embeddingType must be one of classicalmds, nonmetricmds or pca')

    value = _igt_projection_core(data_temporal_map=data_temporal_map, dimensions=dimensions,
                                 embedding_type=embedding_type)
    return value
