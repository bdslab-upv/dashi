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
Multi Source Variability (MSV) metrics estimation module
"""

from dataclasses import dataclass

import numpy as np
from typing import Union, Dict, Optional

from dashi.unsupervised_characterization.utils import _js_divergence, _cmdscale
from dashi.unsupervised_characterization.data_source_map.data_source_map import DataSourceMap, MultiVariateDataSourceMap

__all__ = ['MSVMetrics', 'estimate_MSV_metrics']


@dataclass
class MSVMetrics:
    GPD: Optional[float] = None
    SPO: Optional[np.array] = None
    vertices: Optional[np.array] = None
    sources: Optional[np.array] = None
    nBySource: Optional[np.array] = None

def _distc(n: int) -> float:
    if n == 1:
        return 0.5
    gamma = np.arccos(-1 / n)
    result = np.sin((np.pi - gamma) / 2) / np.sin(gamma)
    return result


def _combine_conditional_source_maps(conditional_maps: Dict[str, Union[DataSourceMap, MultiVariateDataSourceMap]]
                                     ) -> DataSourceMap:
    probability_map_list: list = []
    counts_map_list: list = []
    source_indexes = []
    variable_names = set()
    all_multivariate = all(isinstance(conditional_map, MultiVariateDataSourceMap)
                           for conditional_map in conditional_maps.values())

    for conditional_map in conditional_maps.values():
        probability_map_list.append(conditional_map.probability_map)
        counts_map_list.append(conditional_map.counts_map)
        source_indexes.append(np.array(conditional_map.sources))
        variable_names.add(conditional_map.variable_name)

    sources = np.unique(np.concatenate(source_indexes))
    aligned_probability_maps = []
    aligned_counts_maps = []
    for probability_map, counts_map, source_index in zip(probability_map_list, counts_map_list, source_indexes):
        aligned_probability_maps.append(
            np.array([probability_map[np.where(source_index == source)[0][0]]
                      if source in source_index else np.full(probability_map.shape[1], np.nan)
                      for source in sources])
        )
        aligned_counts_maps.append(
            np.array([counts_map[np.where(source_index == source)[0][0]]
                      if source in source_index else np.full(counts_map.shape[1], np.nan)
                      for source in sources])
        )

    concatenated_matrix = np.concatenate(aligned_probability_maps, axis=1)
    row_sums = np.nansum(concatenated_matrix, axis=1, keepdims=True)
    normalized_matrix = np.full_like(concatenated_matrix, np.nan, dtype=float)

    valid_mask = np.isfinite(row_sums) & (row_sums > 0)
    np.divide(concatenated_matrix, row_sums, out=normalized_matrix, where=valid_mask)

    counts_map = np.nansum(aligned_counts_maps, axis=0)

    variable_name = 'Conditional DSM'
    if not all_multivariate and len(variable_names) == 1:
        variable_name = f'Conditional {next(iter(variable_names))}'

    return DataSourceMap(
        probability_map=normalized_matrix,
        counts_map=counts_map,
        sources=sources,
        support=None,
        variable_name=variable_name,
        variable_type='float64'
    )


def _resolve_source_map_for_msv(
        data_source_map: Union[DataSourceMap, MultiVariateDataSourceMap, Dict],
        variable_name: Optional[str] = None
) -> Union[DataSourceMap, MultiVariateDataSourceMap]:
    if isinstance(data_source_map, (DataSourceMap, MultiVariateDataSourceMap)):
        if variable_name is not None and variable_name != data_source_map.variable_name:
            raise ValueError('variable_name does not match the provided DataSourceMap variable_name.')
        return data_source_map

    if not isinstance(data_source_map, dict):
        raise TypeError('data_source_map must be a DataSourceMap, MultiVariateDataSourceMap, or dictionary.')

    if len(data_source_map) == 0:
        raise ValueError('data_source_map dictionary must contain at least one map.')

    if variable_name is not None and variable_name in data_source_map:
        selected_map = data_source_map[variable_name]
        if not isinstance(selected_map, (DataSourceMap, MultiVariateDataSourceMap)):
            raise TypeError('The selected variable_name entry must be a source map object.')
        return selected_map

    values = list(data_source_map.values())

    if all(isinstance(value, MultiVariateDataSourceMap) for value in values):
        if variable_name is not None:
            raise ValueError('variable_name is only valid for dictionaries of univariate DataSourceMap objects.')
        return _combine_conditional_source_maps(data_source_map)

    if all(isinstance(value, DataSourceMap) for value in values):
        variable_names = {value.variable_name for value in values}
        if variable_name is not None:
            if variable_name not in variable_names:
                raise ValueError('variable_name was not found as a dictionary key or DataSourceMap variable_name.')
            conditional_maps = {
                label: conditional_map
                for label, conditional_map in data_source_map.items()
                if conditional_map.variable_name == variable_name
            }
            if len(conditional_maps) != len(data_source_map):
                raise ValueError('All conditional DataSourceMap objects must match variable_name.')
            return _combine_conditional_source_maps(conditional_maps)

        if len(variable_names) == 1:
            return _combine_conditional_source_maps(data_source_map)

        raise ValueError('variable_name must be provided when data_source_map is a dictionary of univariate maps.')

    if all(isinstance(value, dict) for value in values):
        if variable_name is None:
            raise ValueError('variable_name must be provided for conditional maps with multiple variables per label.')

        conditional_maps = dict()
        for label, maps_by_variable in data_source_map.items():
            if variable_name not in maps_by_variable:
                raise ValueError(f'Variable {variable_name} not found for label {label}.')
            selected_map = maps_by_variable[variable_name]
            if not isinstance(selected_map, DataSourceMap):
                raise TypeError('Selected conditional univariate maps must be DataSourceMap objects.')
            conditional_maps[label] = selected_map

        return _combine_conditional_source_maps(conditional_maps)

    raise TypeError('data_source_map dictionary values must be all source maps or all dictionaries.')

def estimate_MSV_metrics(
        data_source_map: Union[DataSourceMap, MultiVariateDataSourceMap, Dict[str, MultiVariateDataSourceMap],
        Dict[str, DataSourceMap], Dict],
        variable_name: Optional[str] = None,
) -> MSVMetrics:
    """
    Estimate Multi Source Variability (MSV) metrics from a data source map. It can be either a single `DataSourceMap`,
    a `MultiVariateDataSourceMap`, a dictionary containing `{label: MultiVariateDataSourceMap}`, a dictionary
    containing `{variable_name: DataSourceMap}`, or a conditional univariate dictionary containing
    `{label: DataSourceMap}` or `{label: {variable_name: DataSourceMap}}`.

    Parameters
    ----------
    data_source_map : Union[DataSourceMap, MultiVariateDataSourceMap, Dict]
        The data source map to project. This can either be a `DataSourceMap` object
        (result of estimate_univariate_data_source_map), a `MultiVariateDataSourceMap` object
        (result of estimate_multivariate_data_source_map), a dictionary of `DataSourceMap` objects
        (result of estimate_univariate_data_source_map with multiple variables), a dictionary of
        `MultiVariateDataSourceMap` objects where the keys are labels (result of
        estimate_conditional_data_source_map), or a dictionary of conditional univariate maps (result of
        estimate_conditional_univariate_data_source_map).

    variable_name : Optional[str], optional
        Variable to select when `data_source_map` is a dictionary returned by estimate_univariate_data_source_map
        or estimate_conditional_univariate_data_source_map.

    Returns
    -------
    MSVMetrics
        An instance of MSVMetrics containing the GPD, SPO, vertices, sources, and counts by source.
    """

    if data_source_map is None:
        raise ValueError('data_source_map must be provided.')

    data_source_map = _resolve_source_map_for_msv(
        data_source_map=data_source_map,
        variable_name=variable_name
    )

    probability_map = data_source_map.probability_map

    # Number of sources
    ns = len(data_source_map.sources)

    distsM = np.zeros((ns, ns))

    for i in range(ns- 1 ):
        for j in range(i + 1, ns):
            d = np.sqrt(_js_divergence(probability_map[i, :], probability_map[j, :]))
            distsM[i, j] = d
            distsM[j, i] = d

    # Classical MDS to embed in (ns - 1) dimensions
    vertices = _cmdscale(
        d=distsM,
        k=ns - 1
    )

    c = np.sum(vertices, axis=0) / ns
    cc = np.tile(c, ns).reshape((ns, -1), order='C')
    cc2 = vertices - cc

    dc = np.linalg.norm(cc2, axis=1)

    gpdmetric = np.mean(dc) / _distc(ns)
    sposmetrics = dc / (1 - (1 / ns))

    msv = MSVMetrics(
        GPD=gpdmetric,
        SPO=sposmetrics,
        vertices=vertices,
        sources=data_source_map.sources,
        nBySource = data_source_map.counts_map.sum(axis=1)
    )

    return msv

