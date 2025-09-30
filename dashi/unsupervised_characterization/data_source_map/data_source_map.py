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
Data Source Map main module.
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional, List, Union, Dict

from dashi._constants import VALID_TEMPORAL_PERIODS, VALID_TYPES, VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE, \
    VALID_INTEGER_TYPE, VALID_FLOAT_TYPE, \
    VALID_DATE_TYPE, TEMPORAL_PERIOD_WEEK, TEMPORAL_PERIOD_MONTH, TEMPORAL_PERIOD_YEAR, VALID_CONVERSION_STRING_TYPE, \
    MISSING_VALUE, VALID_TYPES_WITHOUT_DATE, VALID_DIM_REDUCTION_TYPES, PCA, MCA, FAMD
from dashi.unsupervised_characterization.utils import (_estimate_absolute_frequencies, _create_supports, _get_types,
                                                       BaseMultiVariateMap, _perform_dimensionality_reduction,
                                                       _scatter_plot, _compute_kde, _normalize_kde, _date_to_numeric)


@dataclass
class DataSourceMap:
    """
    A class that  contains the statistical distributions of data estimated at a
    specific time period. Both relative and absolute frequencies are included

    Attributes
    ----------
    probability_map: Union[List[List[float]], None]
        Numerical matrix representing the probability distribution temporal map (relative frequency).

    counts_map: Union[List[List[int]], None]
        Numerical matrix representing the counts temporal map (absolute frequency).

    sources: Union[List[str], None]
        List of sources (character) from which the data was obtained.

    support: Union[List[str], None]
        Numerical or character matrix representing the support (the value at each bin) of probability_map
        and counts_map.

    variable_name: Union[str, None]
        Name of the variable (character).

    variable_type: Union[str, None]
        Type of the variable (character).

    period: Union[str, None]
        Batching period among 'week', 'month' and 'year'.
    """

    probability_map: Union[List[List[float]], None] = None
    counts_map: Union[List[List[int]], None] = None
    sources: Union[List[str], None] = None
    support: Union[List[str], None] = None
    variable_name: Union[str, None] = None
    variable_type: Union[str, None] = None

    def check (self) -> Union[List[str], bool]:
        """
        Validates the consistency of the DataSourceMap attributes. This method checks for various
        potential issues, such as mismatched dimensions, invalid periods, or unsupported variable types.

        Returns
        -------
        Union[List[str], bool]:
            Returns a list of error messages if any validation fails, otherwise returns True indicating
            the object is valid.
        """
        errors = []

        # Check if the dimensions of probability_map and counts_map match
        if self.probability_map is not None and self.counts_map is not None:
            if (len(self.probability_map) != len(self.counts_map)
                    or any(len(probability_row) != len(count_row) for probability_row, count_row in
                           zip(self.probability_map, self.counts_map))):
                errors.append("the dimensions of probability_map and counts_map do not match")

        # Check if the length of dates matches the rows of probability_map
        if self.sources is not None and self.probability_map is not None:
            if len(self.sources) != len(self.probability_map):
                errors.append("the length of sources must match the rows of probability_map")

        # Check if the length of dates matches the rows of counts_map
        if self.sources is not None and self.counts_map is not None:
            if len(self.sources) != len(self.counts_map):
                errors.append("the length of sources must match the rows of counts_map")

        # Check if the length of support matches the columns of probability_map
        if self.support is not None and self.probability_map is not None:
            if len(self.support) != len(self.probability_map):
                errors.append("the length of support must match the columns of probability_map")

        # Check if the length of support matches the columns of counts_map
        if self.support is not None and self.counts_map is not None:
            if len(self.support) != len(self.counts_map):
                errors.append("the length of support must match the columns of counts_map")

        # Check if period is one of the valid periods
        if self.period is not None and self.period not in VALID_TEMPORAL_PERIODS:
            errors.append(f"period must be one of the following: {', '.join(VALID_TEMPORAL_PERIODS)}")

        # Check if variableType is one of the valid types
        if self.variable_type is not None and self.variable_type not in VALID_TYPES:
            errors.append(f"variable_type must be one of the following: {', '.join(VALID_TYPES)}")

        return errors if errors else True

@dataclass
class MultiVariateDataSourceMap(BaseMultiVariateMap, DataSourceMap):
    """
    A subclass of DataSourceMap representing a multi-variate multi-source data map.
    In addition to the attributes inherited from the DataSourceMap class, this
    class includes additional properties specific to multivariate multi-source data.

    Attributes
    ----------
    multivariate_probability_map: Optional[np.ndarray]
        List of matrices representing the multi-variate probability distribution
        temporal map (relative frequency) for each timestamp.

    multivariate_counts_map: Optional[np.ndarray]
        List of matrices representing the multi-variate counts temporal map (absolute)
        for each timestamp.

    multivariate_support: Optional[np.ndarray]
        List of matrices representing the support (the value at each bin) of the dimensions
        of multivariate_probability_map and multivariate_counts_map.
    """
    def check(self):
        errors = super().check()
        multi_errors = self.check_multivariate()
        if multi_errors is not True:
            if errors is True:
                errors = []
            errors.extend(multi_errors)
        return errors if errors else True


def estimate_univariate_data_source_map(
    data: pd.DataFrame,
    source_column: str,
    supports: Union[Dict, None] = None,
    numeric_smoothing: bool = True,
    numeric_variables_bins: Optional[int] = 100,
    verbose: bool = False
) -> Union[DataSourceMap, Dict[str, DataSourceMap]]:
    """
    Estimates a DataSourceMap object from a DataFrame containing individuals in rows and the variables
    in columns, being one of these columns the analysis source.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the data to be analyzed. Each row represents an individual and each column a variable.

    source_column: str
        Name of the column in the DataFrame that contains the source of the data. This column will be used to group
        the data and estimate the distributions.

    supports: Union[Dict, None], optional
        A dictionary with structure {variable_name: variable_type_name} containing the support
        of the data distributions for each variable. If not provided, it is automatically
        estimated from the data.

    numeric_smoothing: bool, optional
        Logical value indicating whether a Kernel Density Estimation smoothing
        (Gaussian kernel, default bandwidth) is to be applied on numerical variables
        or traditional histogram instead.

    numeric_variables_bins: int
        The number of bins at which to define the frequency/density histogram for numerical
        variables when their support is not provided. 100 as default.

    verbose: bool, optional
        If True, prints additional information about the estimation process. Default is False.

    Returns
    -------
    DataSourceMap
        The DataSourceMap object or a dictionary of DataSourceMap objects depending on the number of
        analysis variables.
"""

    if data is None:
        raise ValueError('An input data frame is required')

    if source_column is None:
        raise ValueError('A source column name is required')

    if source_column not in data.columns:
        raise ValueError(f'Source column "{source_column}" not found in the data frame')

    if not all(data[column].dtype.name in VALID_TYPES for column in data.columns):
        print(data.dtypes)
        raise ValueError(f'The classes of input columns must be one of the following: {", ".join(VALID_TYPES)}')

    if supports is not None and not all(support in VALID_TYPES_WITHOUT_DATE for support in supports):
        raise ValueError(
            f'All the elements provided in the supports parameter must be of type {", ".join(VALID_TYPES_WITHOUT_DATE)}')


    sources = data[source_column]
    data_without_sources = data.drop(columns=[source_column])
    number_of_columns = len(data_without_sources.columns)

    if number_of_columns == 0:
        raise ValueError("Data must contain at least one variable column apart from source.")

    if verbose:
        print(f'Total number of columns to analyze: {number_of_columns}')
        print(f'Number of sources: {len(sources.unique())}')

    # Get VARIABLE types, others will not be allowed
    data_types, columns_by_type = _get_types(
        data=data_without_sources,
        verbose=verbose
    )

    data_without_sources = _date_to_numeric(
        data=data_without_sources,
        columns_by_type=columns_by_type,
        verbose=verbose
    )

    # Implement the logic to fill supports based on the variable type
    # Create supports
    data_without_sources, supports = _create_supports(
        data=data_without_sources,
        supports=supports,
        columns_types=columns_by_type,
        number_of_columns=number_of_columns,
        numeric_variables_bins=numeric_variables_bins,
        verbose=verbose
    )

    posterior_data_types = data_without_sources.dtypes
    results: dict = {}

    for column in data_without_sources.columns:
        if verbose:
            print(f'Estimation of DataSourceMap for variable: {column}')

        grouped = data_without_sources.groupby(sources, observed=False)[column]

        counts_map:list = []

        for source_value, group in grouped:
            map_data = _estimate_absolute_frequencies(
                group,
                varclass=posterior_data_types[column],
                support=supports[column],
                numeric_smoothing=numeric_smoothing
            )
            counts_map.append(map_data)
        counts_map = np.array(counts_map)

        probability_map = np.array([
            arr / arr.sum() if arr.sum() > 0 else np.zeros_like(arr)
            for arr in counts_map
        ])

        if posterior_data_types[column] == VALID_DATE_TYPE:
            support = pd.DataFrame(pd.to_datetime(supports[column]))
        elif posterior_data_types[column] in [VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE]:
            support = pd.DataFrame(supports[column], columns=[column])
        else:
            support = pd.DataFrame(supports[column])

        data_source_map = DataSourceMap(
            probability_map=probability_map,
            counts_map=counts_map,
            sources=np.unique(sources),
            support=support,
            variable_name=column,
            variable_type=posterior_data_types[column]
        )
        results[column] = data_source_map

    if number_of_columns > 1:
        if verbose:
            print('Returning results as a dictionary of DataSourceMap objects for each column')
        return results

    else:
        if verbose:
            print('Returning a single DataSourceMap object for the single column')
        return results[data_without_sources.columns[0]]


def estimate_multivariate_data_source_map(
        data: pd.DataFrame,
        source_column_name: str,
        kde_resolution: int = 10,
        dimensions: int = 2,
        dim_reduction: str = 'PCA',
        scale: bool = True,
        scatter_plot: bool = False,
        verbose: bool = False
) -> MultiVariateDataSourceMap:
    """
    Estimates a MultiVariateDataSourceMap object from a DataFrame containing multiple variables (in columns) with a source
    column indicating the source of the data. The function performs dimensionality reduction on the data (e.g. PCA, MCA, FAMD)
    to handle high-dimensional data and estimates the probability distributions for each source.

    Parameters
    ----------
    data: pd.DataFrame
        A DataFrame where each row represents an individual or data point, and each column represents a
        variable. One column should represent the analysis sources.

    source_column_name: str
        A string indicating the name of the column in the DataFrame that contains the source of the data.

    kde_resolution: int
        The resolution of the grid used for Kernel Density Estimation (KDE). This determines the granularity
        of the KDE grid and how fine or coarse the estimated density maps will be. Default is 10.

    dimensions: int
        The number of dimensions to keep after applying dimensionality reduction (e.g., PCA).
        Default is 2, meaning the data will be projected into a 2D space. The maximum number of dimensions
        available are 3.

    dim_reduction: str
        A dimensionality reduction technique to be used on the data. Default is `PCA` (Principal Component Analysis)
        for numerical data. Other options can include 'MCA' (Multiple Correspondence Analysis) for categorical data or
        'FAMD' (Factor Analysis of Mixed Data) for mixed data. Note: in case of using 'FAMD', numerical variables must be
        in float type. Otherwise they will be treated as categorical.

    scale: bool
        Applicable just when using PCA dimensionality reduction. If true scales the input data using z-score
        normalization. Defaults to `True`.

    scatter_plot: bool
        Whether to generate a scatter plot of the first two principal components of the dimensionality reduction

    verbose: bool
        Whether to display additional information during the process. Defaults to `False`.

    Returns
    -------
    MultiVariateDataSourceMap
        A MultiVariateDataSourceMap object containing the estimated probability distributions for each source,
        along with the multivariate probability maps, counts maps, and supports for each dimension.
    """
    if data is None:
        raise ValueError('An input data frame is required')

    if source_column_name is None:
        raise ValueError('A source column name is required')

    if source_column_name not in data.columns:
        raise ValueError(f'Source column "{source_column_name}" not found in the data frame')

    if not all(data[column].dtype.name in VALID_TYPES for column in data.columns):
        print(data.dtypes)
        raise ValueError(f'The classes of input columns must be one of the following: {", ".join(VALID_TYPES)}')

    if dim_reduction not in VALID_DIM_REDUCTION_TYPES:
        raise ValueError(f'Dimensionality reduction method must be one of the following: {", ".join(VALID_DIM_REDUCTION_TYPES)}')

    if dimensions not in [2, 3]:
        raise ValueError('Dimensions must be either 2 or 3 for the multivariate data source map')

    sources = data[source_column_name]
    data_without_sources = data.drop(columns=[source_column_name])
    number_of_columns = len(data_without_sources.columns)

    if verbose:
        print(f'Total number of columns to analyze: {number_of_columns}')
        print(f'Number of sources: {len(sources.unique())}')

    data_types, columns_by_type = _get_types(
        data=data_without_sources,
        verbose=verbose
    )

    data_without_sources = _date_to_numeric(
        data=data_without_sources,
        columns_by_type=columns_by_type,
        verbose=verbose
    )

    if verbose:
        print(f'Applying dimensionality reduction with {dim_reduction}')

    reduced_data = _perform_dimensionality_reduction(
        data_without_sources,
        dim_reduction=dim_reduction,
        n_components=dimensions,
        verbose=verbose,
        scale=scale
    )

    reduced_data[[source_column_name]] = pd.DataFrame({
        source_column_name: sources
    })

    if scatter_plot:
        _scatter_plot(reduced_data=reduced_data, dim_reduction=dim_reduction, verbose=verbose,
                      color_column=source_column_name)

    value_counts = sources.value_counts(sort=False)

    sources_info = {
        'sources': np.unique(sources),
        'value_counts': value_counts[value_counts > dimensions],
        'source_column_name': source_column_name
    }

    dsm = _generate_multivariate_dsm(
        reduced_data=reduced_data,
        sources_info=sources_info,
        verbose=verbose,
        dimensions=dimensions,
        kde_resolution=kde_resolution
    )

    return dsm

def estimate_conditional_data_source_map(
        data: pd.DataFrame,
        source_column_name: str,
        label_column_name: str,
        kde_resolution: int = 10,
        dimensions: int = 2,
        dim_reduction: str = 'PCA',
        scale: bool = True,
        scatter_plot: bool = False,
        verbose: bool = False
) -> Dict[str, MultiVariateDataSourceMap]:
    """
    Estimates a MultiVariateDataSourceMap object for the data corresponding to each label of a DataFrame
    containing multiple variables (in columns) over different sources, using dimensionality reduction techniques (e.g., PCA)
    to handle high dimensional data.

    Parameters
    ----------
    data: pd.DataFrame
        A DataFrame where each row represents an individual or data point, and each column represents a
        variable. One column should represent the analysis sources and another the labels.

    source_column_name: str
        A string indicating the name of the column in the DataFrame that contains the source of the data.

    label_column_name: str
        The name of the column that contains the labels or class/category for each observation
        (used for conditional analysis).

    kde_resolution: int
        The resolution of the grid used for Kernel Density Estimation (KDE). This determines the granularity
        of the KDE grid and how fine or coarse the estimated density maps will be. Default is 10.

    dimensions: int
        The number of dimensions to keep after applying dimensionality reduction (e.g., PCA).
        Default is 2, meaning the data will be projected into a 2D space. The maximum number of dimensions
        available are 3. For single variable datasets, dimensions can be set to 1

    dim_reduction: str
        A dimensionality reduction technique to be used on the data. Default is 'PCA' (Principal Component Analysis)
        for numerical data. Other options can include 'MCA' (Multiple Correspondence Analysis) for categorical data or
        'FAMD' (Factor Analysis of Mixed Data) for mixed data. Note: in case of using 'FAMD', numerical variables must be
        in float type. Otherwise they will be treated as categorical.

    scale: str
        Applicable just when using PCA dimensionality reduction. If true scales the input data using z-score
        normalization. Defaults to `True`

    scatter_plot: bool
        Whether to generate a scatter plot of the first two principal components of the dimensionality reduction.

    verbose: bool
        Whether to display additional information during the process. Defaults to `False`.

    Returns
    -------
    Dict[str, MultiVariateDataSourceMap]
        A dictionary where the keys are the labels in the dataset, and the values are
        `MultiVariateDataSourceMap` objects representing the multi-source maps generated for each label.
    """
    if data is None:
        raise ValueError('An input data frame is required')

    if source_column_name is None:
        raise ValueError('The name of the column including sources is required.')

    if source_column_name not in data.columns:
        raise ValueError(f'Source column "{source_column_name}" not found in the data frame')

    if not all(data[column].dtype.name in VALID_TYPES for column in data.columns):
        print(data.dtypes)
        raise ValueError(f'The types of input columns must be one of the following: {", ".join(VALID_TYPES)}')

    if dim_reduction not in VALID_DIM_REDUCTION_TYPES:
        raise ValueError(
            f'Dimensionality reduction method must be one of the following: {", ".join(VALID_DIM_REDUCTION_TYPES)}')

    if dimensions not in [1, 2, 3]:
        raise ValueError(
            f'The number of supported dimensions are 1, 2 or 3')

    labels_column = data[label_column_name]
    sources = data[source_column_name]
    data_without_sources = data.drop(columns=[source_column_name, label_column_name])
    number_of_columns = len(data_without_sources.columns)

    if verbose:
        print(f'Total number of columns to analyze: {number_of_columns}')
        print(f'Number of sources: {len(sources.unique())}')

    data_types, columns_by_type = _get_types(
        data=data_without_sources,
        verbose=verbose
    )

    data_without_sources = _date_to_numeric(
        data=data_without_sources,
        columns_by_type=columns_by_type,
        verbose=verbose
    )

    if verbose:
        print(f'Applying dimensionality reduction with {dim_reduction}')

    reduced_data = _perform_dimensionality_reduction(
        data_without_sources,
        dim_reduction=dim_reduction,
        n_components=dimensions,
        verbose=verbose,
        scale=scale
    )

    reduced_data[[label_column_name, source_column_name]] = pd.DataFrame({
        label_column_name: labels_column,
        source_column_name: sources
    })

    if scatter_plot:
        _scatter_plot(reduced_data=reduced_data, dim_reduction=dim_reduction, verbose=verbose,
                      color_column=label_column_name)

    reduced_data_by_label = {
        label: group.drop(columns=[label_column_name]).reset_index(drop=True)
        for label, group in reduced_data.groupby(label_column_name, observed=True)
    }

    concept_maps_dict: dict = {}
    for label, concept_data in reduced_data_by_label.items():
        if verbose:
            print(f'Label: {label}')

        value_counts = concept_data[source_column_name].value_counts(sort=False)
        sources_info = {
            'sources': np.unique(sources),
            'value_counts': value_counts[value_counts > dimensions],
            'source_column_name': source_column_name
        }
        dsm = _generate_multivariate_dsm(
            reduced_data=concept_data,
            sources_info=sources_info,
            verbose=verbose,
            dimensions=dimensions,
            kde_resolution=kde_resolution
        )

        concept_maps_dict[label] = dsm

    return concept_maps_dict


def _generate_multivariate_dsm(
        reduced_data,
        sources_info,
        verbose,
        dimensions,
        kde_resolution
):
    xmin = reduced_data.drop(columns=[sources_info['source_column_name']]).min(axis=0)
    xmax = reduced_data.drop(columns=[sources_info['source_column_name']]).max(axis=0)

    if verbose:
        print('Estimating the data source maps')

    kde_list: list = []
    for source in sources_info['sources']:
        if source in sources_info['value_counts'].index and sources_info['value_counts'][source] > dimensions:
            kde = _compute_kde(
                reduced_data[reduced_data[sources_info['source_column_name']] == source].drop(
                    columns=[sources_info['source_column_name']]
                ),
                xmin[:dimensions], xmax[:dimensions], kde_resolution
            )
            kde_list.append(kde)
        else:
            if verbose:
                print(f'Not enough data for calculating source: {source} probability map')
            kde_shape = (kde_resolution,) * dimensions
            kde = np.full(kde_shape, np.nan)
            kde_list.append(kde)

    probability_map = np.row_stack([_normalize_kde(kde).flatten() for kde in kde_list])
    multivariate_probability_map = [_normalize_kde(kde) for kde in kde_list]
    multivariate_support = [np.linspace(start, stop, kde_resolution) for start, stop in zip(xmin[:dimensions], xmax[:dimensions])]
    non_nan_mask = ~np.isnan(probability_map).any(axis=1)
    non_nan_probability_map = probability_map[non_nan_mask]
    non_nan_counts_map = np.round(non_nan_probability_map * sources_info['value_counts'].values[:, np.newaxis])
    counts_map = np.full(probability_map.shape, np.nan)
    counts_map[non_nan_mask] = non_nan_counts_map
    multivariate_counts_map: list = []
    index = 0
    for prob_map in multivariate_probability_map:
        if np.isnan(prob_map).any():
            multivariate_counts_map.append(prob_map)
        else:
            multivariate_counts_map.append(np.round(prob_map * sources_info['value_counts'].iloc[index]))
            index += 1

    dsm = MultiVariateDataSourceMap(
        probability_map=probability_map,
        multivariate_probability_map=multivariate_probability_map,
        counts_map=counts_map,
        multivariate_counts_map=multivariate_counts_map,
        sources=sources_info['sources'],
        support=pd.DataFrame(range(0, kde_resolution ** dimensions)),
        multivariate_support=multivariate_support,
        variable_name='Dim.reduced.{}D'.format(dimensions),
        variable_type='float64',
    )

    return dsm



