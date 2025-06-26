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
from typing import Union, Dict

from dashi.unsupervised_characterization.utils import _js_divergence, _cmdscale
from dashi.unsupervised_characterization.data_source_map.data_source_map import DataSourceMap, MultiVariateDataSourceMap


@dataclass
class MSVMetrics:
    GPD: float = None
    SPO: np.array = None
    vertices: np.array = None
    sources: np.array = None
    nBySource: np.array = None

def _distc(n: int) -> float:
    if n == 1:
        return 0.5
    gamma = np.arccos(-1 / n)
    result = np.sin((np.pi - gamma) / 2) / np.sin(gamma)
    return result

def estimate_MSV_metrics(
        data_source_map: Union[DataSourceMap, MultiVariateDataSourceMap, Dict[str, MultiVariateDataSourceMap]],
) -> MSVMetrics:
    """
    Estimate Multi Source Variability (MSV) metrics from a data source map. It can be either a single `DataSourceMap`,
    a `MultiVariateDataSourceMap`, or a dictionary with the following structure `{label: MultiVariateDataSourceMap}`.

    Parameters
    ----------
    data_source_map : Union[DataSourceMap, MultiVariateDataSourceMap, Dict[str, MultiVariateDataSourceMap]]
        The data source map to project. This can either be a `DataSourceMap` object
        (result of estimate_univariate_data_source_map), a `MultiVariateDataSourceMap` object
        (result of estimate_multivariate_data_source_map), or a dictionary of `MultiVariateDataSourceMap` objects
        where the keys are the selected labels (result of estimate_conditional_data_source_map).

    Returns
    -------
    MSVMetrics
        An instance of MSVMetrics containing the GPD, SPO, vertices, sources, and counts by source.
    """

    if data_source_map is None:
        raise ValueError('data_source_map must be provided.')

    if isinstance(data_source_map, dict) and all(isinstance(v, MultiVariateDataSourceMap) for v in data_source_map.values()):
        probability_map_list: list = []
        sources_list: list = []
        counts_map_list: list = []

        for label, conditional_map in data_source_map.items():
            probability_map_list.append(conditional_map.probability_map)
            sources_list.append(conditional_map.sources)
            counts_map_list.append(conditional_map.counts_map)
        sources = np.unique(sources_list)

        concatenated_matrix = np.concatenate(probability_map_list, axis=1)
        row_sums = np.nansum(concatenated_matrix, axis=1, keepdims=True)
        normalized_matrix = np.divide(concatenated_matrix, row_sums)

        counts_map = np.nansum(counts_map_list, axis=0)

        data_source_map = DataSourceMap(
            probability_map=normalized_matrix,
            counts_map=counts_map,
            sources=sources,
            support=None,
            variable_name='Conditional DSM',
            variable_type='float64'
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

