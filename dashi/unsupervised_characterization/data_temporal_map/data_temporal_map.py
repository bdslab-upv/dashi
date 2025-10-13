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
Data Temporal Map main functions and classes
"""
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Union, List, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import prince
from scipy.stats import gaussian_kde

from dashi._constants import VALID_TEMPORAL_PERIODS, VALID_TYPES, VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE, \
    VALID_INTEGER_TYPE, VALID_FLOAT_TYPE, \
    VALID_DATE_TYPE, TEMPORAL_PERIOD_WEEK, TEMPORAL_PERIOD_MONTH, TEMPORAL_PERIOD_YEAR, VALID_CONVERSION_STRING_TYPE, \
    MISSING_VALUE, VALID_TYPES_WITHOUT_DATE, VALID_DIM_REDUCTION_TYPES, PCA, MCA, FAMD
from dashi.unsupervised_characterization.utils import (_estimate_absolute_frequencies, _create_supports, _get_types,
                                                       BaseMultiVariateMap, _perform_dimensionality_reduction,
                                                       _scatter_plot, _normalize_kde, _compute_kde)


@dataclass
class DataTemporalMap:
    """
    A class that  contains the statistical distributions of data estimated at a
    specific time period. Both relative and absolute frequencies are included

    Attributes
    ----------
    probability_map: Union[List[List[float]], None]
        Numerical matrix representing the probability distribution temporal map (relative frequency).

    counts_map: Union[List[List[int]], None]
        Numerical matrix representing the counts temporal map (absolute frequency).

    dates: Union[List[datetime], None]
        Array of the temporal batches.

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
    dates: Union[List[datetime], None] = None
    support: Union[List[str], None] = None
    variable_name: Union[str, None] = None
    variable_type: Union[str, None] = None
    period: Union[str, None] = None

    def check(self) -> Union[List[str], bool]:
        """
        Validates the consistency of the DataTemporalMap attributes. This method checks for various
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
        if self.dates is not None and self.probability_map is not None:
            if len(self.dates) != len(self.probability_map):
                errors.append("the length of dates must match the rows of probability_map")

        # Check if the length of dates matches the rows of counts_map
        if self.dates is not None and self.counts_map is not None:
            if len(self.dates) != len(self.counts_map):
                errors.append("the length of dates must match the rows of counts_map")

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
class MultiVariateDataTemporalMap(BaseMultiVariateMap, DataTemporalMap):
    """
    A subclass of DataTemporalMap representing a multi-variate time series data map.
    In addition to the attributes inherited from the DataTemporalMap class, this
    class includes additional properties specific to multivariate time series data.

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


def trim_data_temporal_map(
        data_temporal_map: DataTemporalMap,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
) -> DataTemporalMap:
    """
    Trims the data in the DataTemporalMap object to the specified date range.

    Parameters
    ----------
    data_temporal_map: DataTemporalMap
        The DataTemporalMap object to be trimmed.

    start_date: Optional[datetime]
        The start date of the range to trim the data from. If None, the earliest
        date in `data_temporal_map.dates` will be used.

    end_date: Optional[datetime]
        The end date of the range to trim the data from. If None, the latest
        date in `data_temporal_map.dates` will be used.

    Returns
    -------
    DataTemporalMap:
        The input DataTemporalMap object with trimmed data.
    """
    if start_date is None:
        start_date = data_temporal_map.dates.min()
    else:
        start_date = data_temporal_map.dates[data_temporal_map.dates >= start_date].min()

    if end_date is None:
        end_date = data_temporal_map.dates.max()
    else:
        end_date = data_temporal_map.dates[data_temporal_map.dates <= end_date].max()

    start_index = data_temporal_map.dates.get_loc(start_date)
    end_index = data_temporal_map.dates.get_loc(end_date) + 1

    data_temporal_map.probability_map = data_temporal_map.probability_map[start_index:end_index]
    data_temporal_map.counts_map = data_temporal_map.counts_map[start_index:end_index]
    data_temporal_map.dates = data_temporal_map.dates[start_index:end_index]

    return data_temporal_map


def estimate_univariate_data_temporal_map(
        data: pd.DataFrame,
        date_column_name: str,
        period: str = TEMPORAL_PERIOD_MONTH,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        supports: Union[Dict, None] = None,
        numeric_variables_bins: int = 100,
        numeric_smoothing: bool = True,
        date_gaps_smoothing: bool = False,
        verbose: bool = False
) -> Union[DataTemporalMap, Dict[str, DataTemporalMap]]:
    """
    Estimates a DataTemporalMap object from a DataFrame containing individuals in rows and the variables
    in columns, being one of these columns the analysis date (typically the acquisition date).

    Parameters
    ----------
    data: pd.DataFrame
        A DataFrame containing as many rows as individuals, and as many columns as teh analysis
        variables plus the individual acquisition date.

    date_column_name: str
        A string indicating teh name of the column in data containing the analysis date variable.

    period:
        The period to batch the data for analysis. Options are:
        - 'week' (weekly analysis)
        - 'month' (monthly analysis, default)
        - 'year' (annual analysis)

    start_date: pd.Timestamp
        A date object indicating the date at which to start teh analysis, in case of being different
        from the first chronological date in the date column.

    end_date: pd.Timestamp
        A date object indicating the date at which to end the analysis, in case of being
        different from the last chronological date in the date column.

    supports: Union[Dict, None], optional
        A dictionary with structure {variable_name: variable_type_name} containing the support
        of the data distributions for each variable. If not provided, it is automatically
        estimated from the data.

    numeric_variables_bins: int
        The number of bins at which to define the frequency/density histogram for numerical
        variables when their support is not provided. 100 as default.

    numeric_smoothing: bool
        Logical value indicating whether a Kernel Density Estimation smoothing
        (Gaussian kernel, default bandwidth) is to be applied on numerical variables
        or traditional histogram instead.

    date_gaps_smoothing: bool
        Logical value indicating whether a linear smoothing is applied to those time
        batches without data. By default, gaps are filled with NAs.

    verbose: bool
        If True, prints additional information about the estimation process. Default is False.

    Returns
    -------
    DataTemporalMap
        The DataTemporalMap object or a dictionary of DataTemporalMap objects depending on the number of
        analysis variables.
    """
    # Validation of parameters
    if data is None:
        raise ValueError('An input data frame is required.')

    if len(data.columns) < 2:
        raise ValueError('An input data frame is required with at least 2 columns, one for dates.')

    if date_column_name is None:
        raise ValueError('The name of the column including dates is required.')

    if date_column_name not in data.columns:
        raise ValueError(f'There is not a column named \'{date_column_name}\' in the input data.')

    if data[date_column_name].dtype != VALID_DATE_TYPE:
        raise ValueError('The specified date column must be of type pandas.Timestamp.')

    if period not in VALID_TEMPORAL_PERIODS:
        raise ValueError(f'Period must be one of the following: {", ".join(VALID_TEMPORAL_PERIODS)}')

    if not all(data[column].dtype.name in VALID_TYPES for column in data.columns):
        print(data.dtypes)
        raise ValueError(f'The classes of input columns must be one of the following: {", ".join(VALID_TYPES)}')

    if start_date is not None and not isinstance(start_date, pd.Timestamp):
        raise ValueError('The specified start date must be of type pandas.Timestamp')

    if end_date is not None and not isinstance(end_date, pd.Timestamp):
        raise ValueError('The specified end date must be of type pandas.Timestamp')

    if supports is not None and not all(support in VALID_TYPES_WITHOUT_DATE for support in supports):
        raise ValueError(
            f'All the elements provided in the supports parameter must be of type {", ".join(VALID_TYPES_WITHOUT_DATE)}')

    # Separate analysis data from analysis dates
    dates = data[date_column_name]
    data_without_date_column = data.drop(columns=[date_column_name])
    number_of_columns = len(data_without_date_column.columns)

    if verbose:
        print(f'Total number of columns to analyze: {number_of_columns}')
        print(f'Analysis period: {period}')

    # Floor the dates to the specified unit
    if period == TEMPORAL_PERIOD_WEEK:
        # Adjust the dates to the beginning of the week (assuming week starts on Sunday)
        dates = dates - pd.to_timedelta((dates.dt.dayofweek + 1) % 7, unit='D')
    elif period == TEMPORAL_PERIOD_MONTH:
        # Adjust the dates to the beginning of the month
        dates = dates - pd.to_timedelta(dates.dt.day - 1, unit='D')
    elif period == TEMPORAL_PERIOD_YEAR:
        # Adjust the dates to the beginning of the year
        dates = dates - pd.to_timedelta(dates.dt.dayofyear - 1, unit='D')

    data_types, columns_by_type = _get_types(
        data=data_without_date_column,
        verbose=verbose
    )

    # Convert dates to numbers
    if any(columns_by_type['date']):
        data_without_date_column.iloc[:, columns_by_type['date']] = data_without_date_column.iloc[:, columns_by_type['date']].apply(
            pd.to_numeric,
            errors='coerce'
        )
        if verbose:
            print('Converting date columns to numeric for distribution analysis')

    data_without_date_column, supports = _create_supports(
        data=data_without_date_column,
        supports=supports,
        columns_types=columns_by_type,
        number_of_columns=number_of_columns,
        numeric_variables_bins=numeric_variables_bins,
        verbose=verbose
    )

    # Estimate the Data Temporal Map
    posterior_data_classes = data_without_date_column.dtypes
    results = {}

    if verbose:
        print('Estimating the data temporal maps')

    for column in data_without_date_column.columns:
        if verbose:
            print(f'Estimating the DataTemporalMap of variable \'{column}\'')

        data_xts = pd.Series(data_without_date_column[column].values, index=pd.to_datetime(dates))
        data_xts = data_xts.sort_index(ascending=True)

        if start_date is not None or end_date is not None:
            if start_date is None:
                start_date = min(dates)
            if end_date is None:
                end_date = max(dates)

            data_xts = data_xts[start_date:end_date]

        period_function = {
            TEMPORAL_PERIOD_WEEK: data_xts.resample('W').apply(
                _estimate_absolute_frequencies,
                varclass=posterior_data_classes[column],
                support=supports[column],
                numeric_smoothing=numeric_smoothing
            ),
            TEMPORAL_PERIOD_MONTH: data_xts.resample('MS').apply(
                _estimate_absolute_frequencies,
                varclass=posterior_data_classes[column],
                support=supports[column],
                numeric_smoothing=numeric_smoothing
            ),
            TEMPORAL_PERIOD_YEAR: data_xts.resample('YS').apply(
                _estimate_absolute_frequencies,
                varclass=posterior_data_classes[column],
                support=supports[column],
                numeric_smoothing=numeric_smoothing
            )
        }
        mapped_data = pd.DataFrame(period_function[period].tolist(), period_function[period].index)
        dates_map = pd.to_datetime(mapped_data.index)

        sequence_date_period = None

        if period == TEMPORAL_PERIOD_WEEK:
            sequence_date_period = 'W'
        elif period == TEMPORAL_PERIOD_MONTH:
            sequence_date_period = 'MS'
        elif period == TEMPORAL_PERIOD_YEAR:
            sequence_date_period = 'YS'

        full_date_sequence = pd.date_range(min(dates_map), max(dates_map), freq=sequence_date_period)
        date_gaps_smoothing_done = False

        if len(dates_map) != len(full_date_sequence):
            number_of_gaps = len(full_date_sequence) - len(dates_map)

            data_series_sequence = pd.Series(index=full_date_sequence)
            mapped_data = pd.concat([mapped_data, data_series_sequence], axis=1)

            if date_gaps_smoothing:
                mapped_data.interpolate(method='linear', axis=0, inplace=True)
                if verbose:
                    print(f'-\'{column}\': {number_of_gaps} {period} date gaps filled by linear smoothing')
                    date_gaps_smoothing_done = True
            else:
                if verbose:
                    print(f'-\'{column}\': {number_of_gaps} {period} date gaps filled by NAs')

            dates_map = pd.to_datetime(mapped_data.index)
        else:
            if verbose and date_gaps_smoothing:
                print(f'-\'{column}\': no date gaps, date gap smoothing was not applied')

        counts_map = mapped_data.values

        probability_arrays = []
        for array in counts_map:
            probability_arrays.append(
                np.divide(array, array.sum())
            )
        probability_map = np.array(probability_arrays)

        if posterior_data_classes[column] == VALID_DATE_TYPE:
            support = pd.DataFrame(pd.to_datetime(supports[column]))
        elif posterior_data_classes[column] in [VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE]:
            support = pd.DataFrame(supports[column], columns=[column])
        else:
            support = pd.DataFrame(supports[column])

        if date_gaps_smoothing_done and np.any(np.isnan(probability_map)):
            print(
                f'Date gaps smoothing was performed in \'{column}\' variable but some gaps will still be reflected in the resultant probabilityMap (this is generally due to temporal heatmap sparsity)')

        data_temporal_map = DataTemporalMap(
            probability_map=probability_map,
            counts_map=counts_map,
            dates=dates_map,
            support=support,
            variable_name=column,
            variable_type=posterior_data_classes[column],
            period=period
        )
        results[column] = data_temporal_map

    if number_of_columns > 1:
        if verbose:
            print('Returning results as a dictionary of DataTemporalMap objects')
        return results
    else:
        if verbose:
            print('Returning results as an individual DataTemporalMap object')
        return results[data_without_date_column.columns[0]]

def estimate_multivariate_data_temporal_map(
        data: pd.DataFrame,
        date_column_name: str,
        kde_resolution: int = 10,
        dimensions: int = 2,
        period: str = TEMPORAL_PERIOD_MONTH,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        dim_reduction: str = 'PCA',
        scale: bool = True,
        scatter_plot: bool = False,
        verbose: bool = False
) -> MultiVariateDataTemporalMap:
    """
    Estimates a MultiVariateDataTemporalMap object from a DataFrame containing multiple variables
    (in columns) over time, using dimensionality reduction techniques (e.g., PCA) to handle high-dimensional data.

    Parameters
    ----------
    data: pd.DataFrame
        A DataFrame where each row represents an individual or data point, and each column represents a
        variable. One column should represent the analysis date (typically the acquisition date).

    date_column_name: str
        A string indicating the name of the column in data containing the analysis date variable.

    kde_resolution: int
        The resolution of the grid used for Kernel Density Estimation (KDE). This determines the granularity
        of the KDE grid and how fine or coarse the estimated density maps will be. Default is 10.

    dimensions: int
        The number of dimensions to keep after applying dimensionality reduction (e.g., PCA).
        Default is 2, meaning the data will be projected into a 2D space. The maximum number of dimensions
        available are 3.

    period: str
        The period to batch the data for analysis. Options are:
        - 'week' (weekly analysis)
        - 'month' (monthly analysis, default)
        - 'year' (annual analysis)

    start_date: pd.Timestamp
        A date object indicating the date at which to start teh analysis, in case of being different
        from the first chronological date in the date column.

    end_date: pd.Timestamp
        A date object indicating the date at which to end the analysis, in case of being
        different from the last chronological date in the date column.

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
    MultiVariateDataTemporalMap
        A MultiVariateDataTemporalMap object containing the estimated probability and counts maps for each time period.
    """
    # Validation of parameters
    if data is None:
        raise ValueError('An input data frame is required.')

    if period not in VALID_TEMPORAL_PERIODS:
        raise ValueError(f'Period must be one of the following: {", ".join(VALID_TEMPORAL_PERIODS)}')

    if date_column_name is None:
        raise ValueError('The name of the column including dates is required.')

    if date_column_name not in data.columns:
        raise ValueError(f'There is not a column named \'{date_column_name}\' in the input data.')

    if data[date_column_name].dtype != VALID_DATE_TYPE:
        raise ValueError('The specified date column must be of type pandas.Timestamp.')

    if period not in VALID_TEMPORAL_PERIODS:
        raise ValueError(f'Period must be one of the following: {", ".join(VALID_TEMPORAL_PERIODS)}')

    if not all(data[column].dtype.name in VALID_TYPES for column in data.columns):
        print(data.dtypes)
        raise ValueError(f'The types of input columns must be one of the following: {", ".join(VALID_TYPES)}')

    if start_date is not None and not isinstance(start_date, pd.Timestamp):
        raise ValueError('The specified start date must be of type pandas.Timestamp')

    if end_date is not None and not isinstance(end_date, pd.Timestamp):
        raise ValueError('The specified end date must be of type pandas.Timestamp')

    if dim_reduction not in VALID_DIM_REDUCTION_TYPES:
        raise ValueError(
            f'Dimensionality reduction method must be one of the following: {", ".join(VALID_DIM_REDUCTION_TYPES)}')

    if dimensions not in [2, 3]:
        raise ValueError(
            f'The number of supported dimensions are 2 or 3')

    # Separate analysis data from analysis dates
    dates = data[date_column_name]
    data_without_date_column = data.drop(columns=[date_column_name])
    number_of_columns = len(data_without_date_column.columns)

    if start_date is not None or end_date is not None:
        data_without_date_column = data_without_date_column.set_index(dates)
        data_without_date_column = data_without_date_column.sort_index(ascending=True)
        if start_date is None:
            start_date = min(dates)
        if end_date is None:
            end_date = max(dates)
        data_without_date_column = data_without_date_column.loc[start_date:end_date]
        dates = pd.Series(data_without_date_column.index)
        data_without_date_column = data_without_date_column.reset_index(drop=True)

    if period == TEMPORAL_PERIOD_MONTH:
        dates_for_batching = pd.to_datetime(dates).dt.to_period('M').astype('str')
        full_range = pd.date_range(start=dates_for_batching.min(), end=dates_for_batching.max(), freq='MS')
        unique_dates = pd.to_datetime(full_range)
    if period == TEMPORAL_PERIOD_YEAR:
        dates_for_batching = pd.to_datetime(dates).dt.to_period('Y').astype('str')
        full_range = pd.date_range(start=dates_for_batching.min(), end=dates_for_batching.max(), freq='YS')
        unique_dates = pd.to_datetime(full_range)
    if period == TEMPORAL_PERIOD_WEEK:
        dates_for_batching = pd.to_datetime(dates).dt.to_period('W').astype('str')
        full_range = pd.date_range(start=dates_for_batching.min(), end=dates_for_batching.max(), freq='W-SUN')
        unique_dates = pd.to_datetime(full_range)

    if verbose:
        print(f'Total number of columns to analyze: {number_of_columns}')
        print(f'Analysis period: {period}')

    data_types, columns_by_type = _get_types(
        data=data_without_date_column,
        verbose=verbose
    )

    # Convert dates to numbers
    if any(columns_by_type['date']):
        data_without_date_column.iloc[:, columns_by_type['date']] = data_without_date_column.iloc[:, columns_by_type['date']].apply(
            pd.to_numeric,
            errors='coerce'
        )
        if verbose:
            print('Converting date columns to numeric for distribution analysis')

    if verbose:
        print(f'Applying dimensionality reduction with {dim_reduction}')

    reduced_data = _perform_dimensionality_reduction(
        data_without_date_column,
        dim_reduction=dim_reduction,
        n_components=dimensions,
        verbose=verbose,
        scale=scale
    )

    reduced_data[[date_column_name]] = pd.DataFrame({
        date_column_name: pd.to_datetime(dates_for_batching)
    })

    if scatter_plot:
        _scatter_plot(reduced_data=reduced_data, dim_reduction=dim_reduction, verbose=verbose,
                      color_column=date_column_name)

    value_counts = reduced_data[date_column_name].value_counts(sort=False)
    dates_info = {
        'period': period,
        'unique_dates': unique_dates,
        'value_counts': value_counts[value_counts > dimensions],
        'date_column_name': date_column_name
    }

    dtm = _generate_multivariate_dtm(reduced_data=reduced_data, dates_info=dates_info, verbose=verbose,
                                     dimensions=dimensions, kde_resolution=kde_resolution)
    return dtm


def estimate_conditional_data_temporal_map(
        data: pd.DataFrame,
        date_column_name: str,
        label_column_name: str,
        kde_resolution: int = 10,
        dimensions: int = 2,
        period: str = 'month',
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        dim_reduction: str = 'PCA',
        scale: bool = True,
        scatter_plot: bool = False,
        verbose: bool = False
) -> Dict[str, MultiVariateDataTemporalMap]:
    """
    Estimates a MultivariateDataTemporalMap object for the data corresponding to each label of a DataFrame
    containing multiple variables (in columns) over time, using dimensionality reduction techniques (e.g., PCA)
    to handle high dimensional data.

    Parameters
    ----------
    data: pd.DataFrame
        A DataFrame where each row represents an individual or data point, and each column represents a
        variable. One column should represent the analysis date (typically the acquisition date).

    date_column_name: str
        A string indicating the name of the column in data containing the analysis date variable.

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

    period: str
        The period to batch the data for analysis. Options are:
        - 'week' (weekly analysis)
        - 'month' (monthly analysis, default)
        - 'year' (annual analysis)

    start_date: pd.Timestamp
        A date object indicating the date at which to start teh analysis, in case of being different
        from the first chronological date in the date column.

    end_date: pd.Timestamp
        A date object indicating the date at which to end the analysis, in case of being
        different from the last chronological date in the date column.

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
    Dict[str, MultiVariateDataTemporalMap]
        A dictionary where the keys are the labels in the dataset, and the values are
        `MultiVariateDataTemporalMap` objects representing the temporal maps generated for each label.
    """

    # Validation of parameters
    if data is None:
        raise ValueError('An input data frame is required.')

    if period not in VALID_TEMPORAL_PERIODS:
        raise ValueError(f'Period must be one of the following: {", ".join(VALID_TEMPORAL_PERIODS)}')

    if date_column_name is None:
        raise ValueError('The name of the column including dates is required.')

    if date_column_name not in data.columns:
        raise ValueError(f'There is not a column named \'{date_column_name}\' in the input data.')

    if data[date_column_name].dtype != VALID_DATE_TYPE:
        raise ValueError('The specified date column must be of type pandas.Timestamp.')

    if period not in VALID_TEMPORAL_PERIODS:
        raise ValueError(f'Period must be one of the following: {", ".join(VALID_TEMPORAL_PERIODS)}')

    if not all(data[column].dtype.name in VALID_TYPES for column in data.columns):
        print(data.dtypes)
        raise ValueError(f'The types of input columns must be one of the following: {", ".join(VALID_TYPES)}')

    if start_date is not None and not isinstance(start_date, pd.Timestamp):
        raise ValueError('The specified start date must be of type pandas.Timestamp')

    if end_date is not None and not isinstance(end_date, pd.Timestamp):
        raise ValueError('The specified end date must be of type pandas.Timestamp')

    if dim_reduction not in VALID_DIM_REDUCTION_TYPES:
        raise ValueError(
            f'Dimensionality reduction method must be one of the following: {", ".join(VALID_DIM_REDUCTION_TYPES)}')

    if dimensions not in [1, 2, 3]:
        raise ValueError(
            f'The number of supported dimensions are 1, 2 or 3')

    if dimensions == 1 and scatter_plot:
        raise ValueError('Scatter plot cannot be generated when dimensions is set to 1')

    # Separate analysis data from analysis dates
    labels_columns = data[label_column_name]
    dates = data[date_column_name]
    data_without_date_column = data.drop(columns=[date_column_name, label_column_name])
    number_of_columns = len(data_without_date_column.columns)

    if start_date is not None or end_date is not None:
        data_without_date_column = data_without_date_column.set_index(dates)
        data_without_date_column = data_without_date_column.sort_index(ascending=True)
        if start_date is None:
            start_date = min(dates)
        if end_date is None:
            end_date = max(dates)
        data_without_date_column = data_without_date_column.loc[start_date:end_date]
        dates = pd.Series(data_without_date_column.index)
        data_without_date_column = data_without_date_column.reset_index(drop=True)

    if period == TEMPORAL_PERIOD_MONTH:
        dates_for_batching = pd.to_datetime(dates).dt.to_period('M').astype('str')
        full_range = pd.date_range(start=dates_for_batching.min(), end=dates_for_batching.max(), freq='MS')
        unique_dates = pd.to_datetime(full_range)
    if period == TEMPORAL_PERIOD_YEAR:
        dates_for_batching = pd.to_datetime(dates).dt.to_period('Y').astype('str')
        full_range = pd.date_range(start=dates_for_batching.min(), end=dates_for_batching.max(), freq='YS')
        unique_dates = pd.to_datetime(full_range)
    if period == TEMPORAL_PERIOD_WEEK:
        dates_for_batching = pd.to_datetime(dates).dt.to_period('W').astype('str')
        full_range = pd.date_range(start=dates_for_batching.min(), end=dates_for_batching.max(), freq='W-SUN')
        unique_dates = pd.to_datetime(full_range)

    if verbose:
        print(f'Total number of columns to analyze: {number_of_columns}')
        print(f'Analysis period: {period}')

    data_types, columns_by_type = _get_types(
        data=data_without_date_column,
        verbose=verbose
    )

    # Convert dates to numbers
    if any(columns_by_type['date']):
        data_without_date_column.iloc[:, columns_by_type['date']] = data_without_date_column.iloc[:, columns_by_type['date']].apply(
            pd.to_numeric,
            errors='coerce'
        )
        if verbose:
            print('Converting date columns to numeric for distribution analysis')

    # Dimensionality reduction
    if verbose:
        print(f'Applying dimensionality reduction with {dim_reduction}')

    reduced_data = _perform_dimensionality_reduction(
        data_without_date_column,
        dim_reduction=dim_reduction,
        n_components=dimensions,
        verbose=verbose,
        scale=scale
    )

    reduced_data[[label_column_name, date_column_name]] = pd.DataFrame({
        label_column_name: labels_columns,
        date_column_name: pd.to_datetime(dates_for_batching)
    })

    if scatter_plot:
            _scatter_plot(reduced_data=reduced_data, dim_reduction=dim_reduction, verbose=verbose,
                      color_column=label_column_name)

    reduced_data_by_label = {
        label: group.drop(columns=[label_column_name]).reset_index(drop=True)
        for label, group in reduced_data.groupby(label_column_name, observed=True)
    }

    # Generate DTMs
    concept_maps_dict: dict = {}
    for label, concept_data in reduced_data_by_label.items():
        if verbose:
            print(f'Label :{label}')

        value_counts = concept_data[date_column_name].value_counts(sort=False)
        dates_info = {
            'period': period,
            'unique_dates': unique_dates,
            'value_counts': value_counts[value_counts > dimensions],
            'date_column_name': date_column_name
        }
        dtm = _generate_multivariate_dtm(reduced_data=concept_data, dates_info=dates_info,
                                         verbose=verbose, dimensions=dimensions, kde_resolution=kde_resolution)
        concept_maps_dict[label] = dtm

    return concept_maps_dict


def _generate_multivariate_dtm(reduced_data, dates_info, verbose, dimensions, kde_resolution):
    """
    Generates a MultiVariateDataTemporalMap object from the reduced multivariate data by applying Kernel
    Density Estimation (KDE) of the data over time. This function processes the data in
    the specified temporal period (e.g., weekly, monthly, yearly) and computes the joint probability distribution
    of the multivariate time series.
    """
    xmin = reduced_data.drop(columns=dates_info['date_column_name']).min(axis=0)
    xmax = reduced_data.drop(columns=dates_info['date_column_name']).max(axis=0)

    if verbose:
        print('Estimating the data temporal maps')

    kde_list: list = []
    for date in dates_info['unique_dates']:
        if date in dates_info['value_counts'].index and dates_info['value_counts'][date] > dimensions:
            kde = _compute_kde(reduced_data[reduced_data[dates_info['date_column_name']] == date].drop(
                columns=[dates_info['date_column_name']]),
                xmin[:dimensions], xmax[:dimensions], kde_resolution)
            kde_list.append(kde)
        else:
            if verbose:
                print(f'Not enough data for calculating {date} probability map.')
            kde_shape = (kde_resolution,) * dimensions
            kde = np.full(kde_shape, np.nan)
            kde_list.append(kde)

    probability_map = np.row_stack([_normalize_kde(kde).flatten() for kde in kde_list])
    multivariate_probability_map = [_normalize_kde(kde) for kde in kde_list]
    multivariate_support = [np.linspace(start, stop, kde_resolution) for start, stop in zip(xmin[:dimensions], xmax[:dimensions])]
    non_nan_mask = ~np.isnan(probability_map).any(axis=1)
    non_nan_probability_map = probability_map[non_nan_mask]
    non_nan_counts_map = np.round(non_nan_probability_map * dates_info['value_counts'].values[:, np.newaxis])
    counts_map = np.full(probability_map.shape, np.nan)
    counts_map[non_nan_mask] = non_nan_counts_map
    multivariate_counts_map: list = []
    index = 0
    for prob_map in multivariate_probability_map:
        if np.isnan(prob_map).any():
            multivariate_counts_map.append(prob_map)
        else:
            multivariate_counts_map.append(np.round(prob_map * dates_info['value_counts'].iloc[index]))
            index += 1

    dtm = MultiVariateDataTemporalMap(
        probability_map=probability_map,
        multivariate_probability_map=multivariate_probability_map,
        counts_map=counts_map,
        multivariate_counts_map=multivariate_counts_map,
        dates=dates_info['unique_dates'],
        support=pd.DataFrame(range(0, kde_resolution ** dimensions)),
        multivariate_support=multivariate_support,
        variable_name='Dim.reduced.{}D'.format(dimensions),
        variable_type='float64',
        period=dates_info['period']
    )

    return dtm


