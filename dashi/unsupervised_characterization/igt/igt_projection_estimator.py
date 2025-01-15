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
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

from dashi.unsupervised_characterization.data_temporal_map.data_temporal_map import (trim_data_temporal_map,
                                                                                     DataTemporalMap,
                                                                                     MultiVariateDataTemporalMap)
from dashi.unsupervised_characterization.igt.igt_projection import IGTProjection


def _js_divergence(p, q, epsilon=1e-10):
    """
    Computes the Jensen-Shannon (JS) divergence between two probability distributions.

    The Jensen-Shannon divergence is a symmetric and smoothed version of the Kullback-Leibler (KL) divergence
    and measures the similarity between two probability distributions. Unlike the KL divergence, which is asymmetric,
    the JS divergence is symmetric and bounded, making it a more stable measure for comparing distributions.

    The JS divergence is calculated as:
    JS(p || q) = 0.5 * (KL(p || m) + KL(q || m))
    where m = 0.5 * (p + q) is the average distribution, and KL(p || m) is the Kullback-Leibler divergence between
    distribution `p` and `m`.
    """
    p = np.asarray(p)
    q = np.asarray(q)

    p = np.where(p < epsilon, epsilon, p)
    q = np.where(q < epsilon, epsilon, q)

    m = 0.5 * (p + q)

    kl_p_m = np.where(p != 0, p * np.log2(p / m), 0)
    kl_q_m = np.where(q != 0, q * np.log2(q / m), 0)

    result = 0.5 * (np.nansum(kl_p_m) + np.nansum(kl_q_m))

    return result


def _cmdscale(d, k=2, eig=False, add=False, x_ret=False):
    """
    Performs Classical Multidimensional Scaling (MDS) on a distance matrix to reduce the dimensionality
    of the data, while preserving pairwise distances as much as possible.
    """
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


def _igt_projection_core(data_temporal_map=None, dimensions=3, embedding_type='classicalmds'):
    """
    Computes the core Information Geometric Temporal (IGT) projection for a given DataTemporalMap or
    MultiVariateDataTemporalMap.
    """
    dates = data_temporal_map.dates
    temporal_map = data_temporal_map.probability_map
    number_of_dates = len(dates)

    dissimilarity_matrix = np.zeros((number_of_dates, number_of_dates))
    for i in range(number_of_dates - 1):
        for j in range(i + 1, number_of_dates):
            dissimilarity_matrix[i, j] = np.sqrt(_js_divergence(temporal_map[i, :], temporal_map[j, :]))
            dissimilarity_matrix[j, i] = dissimilarity_matrix[i, j]

    embedding_results = None
    stress_value = None
    if embedding_type == 'classicalmds':
        mds = _cmdscale(dissimilarity_matrix, k=dimensions)

        embedding_results = mds
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

    igt_projection = IGTProjection(
        data_temporal_map=data_temporal_map,
        projection=embedding_results,
        embedding_type=embedding_type,
        stress=stress_value
    )

    return igt_projection


def estimate_igt_projection(data_temporal_map: Union[DataTemporalMap, MultiVariateDataTemporalMap,
                            Dict[str, MultiVariateDataTemporalMap]],
                            dimensions: int = 2,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            embedding_type: str = 'classicalmds'
                            ) -> IGTProjection:
    """
    Estimates the Information Geometric Temporal (IGT) projection of a temporal data map, either a
    `DataTemporalMap`, `MultiVariateDataTemporalMap`, or a dictionary containing
    `{label: MultiVariateDataTemporalMap}`.

    The IGT projection is a technique to visualize the temporal relationships between data batches
    by projecting the data into a lower-dimensional space (e.g., 2D or 3D), with time batches represented
    as points. The distance between points reflects the probabilistic distance between the data distributions
    of those time batches.

    Parameters
    ----------
    data_temporal_map : Union[DataTemporalMap, MultiVariateDataTemporalMap, Dict[str, MultiVariateDataTemporalMap]]
        The temporal data map to project. This can either be a `DataTemporalMap` (result of estimate_data_temporal_map),
        a `MultiVariateDataTemporalMap` (result of estimate_multidim_data_temporal_map), or a dictionary
        of `MultiVariateDataTemporalMap` objects where the keys are labels (result of estimate_multidim_concept_shift).

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

    Returns
    -------
    IGTProjection
        The estimated IGT projection.
    """
    if data_temporal_map is None:
        raise ValueError('dataTemporalMap must be provided')

    if isinstance(data_temporal_map, dict) and all(
            isinstance(value, MultiVariateDataTemporalMap) for value in data_temporal_map.values()):
        probability_maps_list = list()
        dates_list = list()
        for label, concept_map in data_temporal_map.items():
            probability_maps_list.append(concept_map.probability_map)
            dates_list.append(concept_map.dates)
            period = concept_map.period
        dates = pd.to_datetime(np.unique(dates_list))

        # Concatenate the probability maps and normalize
        concatenated_matrix = np.concatenate(probability_maps_list, axis=1)
        row_sums = np.nansum(concatenated_matrix, axis=1, keepdims=True)
        normalized_matrix = np.divide(concatenated_matrix, row_sums)

        data_temporal_map = DataTemporalMap(
            probability_map=normalized_matrix,
            counts_map=None,
            dates=dates,
            support=None,
            variable_name='Concept shift DTM',
            variable_type='float64',
            period=period
        )

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

    if embedding_type not in ['classicalmds', 'nonmetricmds', 'pca']:
        raise ValueError('embeddingType must be one of classicalmds, nonmetricmds or pca')

    value = _igt_projection_core(data_temporal_map=data_temporal_map, dimensions=dimensions,
                                 embedding_type=embedding_type)
    return value
