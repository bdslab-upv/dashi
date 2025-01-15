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

import prince

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import gaussian_kde

from dashi._constants import VALID_TEMPORAL_PERIODS, VALID_TYPES, VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE, \
    VALID_INTEGER_TYPE, VALID_FLOAT_TYPE, \
    VALID_DATE_TYPE, TEMPORAL_PERIOD_WEEK, TEMPORAL_PERIOD_MONTH, TEMPORAL_PERIOD_YEAR, VALID_CONVERSION_STRING_TYPE, \
    MISSING_VALUE, VALID_TYPES_WITHOUT_DATE, VALID_DIM_REDUCTION_TYPES, PCA, MCA, FAMD


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
class MultiVariateDataTemporalMap(DataTemporalMap):
    """
    A subclass of DataTemporalMap representing a multi-variate time series data map.
    In addition to the attributes inherited from the DataTemporalMap class, this
    class includes additional properties specific to multivariate time series data.

    Attributes
    ----------
    multivariate_probability_map: Optional[List[List[float]]]
        List of matrices representing the multi-variate probability distribution
        temporal map (relative frequency) for each timestamp.

    multivariate_counts_map: Optional[List[List[float]]]
        List of matrices representing the multi-variate counts temporal map (absolute)
        for each timestamp.

    multivariate_support: Optional[List[float]]
        List of matrices representing the support (the value at each bin) of the dimensions
        of multivariate_probability_map and multivariate_counts_map.
    """
    multivariate_probability_map: Optional[List[List[float]]] = None
    multivariate_counts_map: Optional[List[List[float]]] = None
    multivariate_support: Optional[List[str]] = None

    def check(self) -> Union[List[str], bool]:
        """
        Validates the consistency of the MultiVariateDataTemporalMap attributes, ensuring
        that the multivariate probability map, counts map, and support dimensions are consistent,
        along with inherited checks from the parent class DataTemporalMap.

        Returns
        -------
        Union[List[str], bool]:
            Returns a list of error messages if any validation fails, otherwise returns True indicating
            the object is valid.
        """
        errors = super().check() if isinstance(super(), DataTemporalMap) else []

        # Check if the dimensions of multivariate_probability_map and multivariate_counts_map match
        if self.multivariate_probability_map is not None and self.multivariate_counts_map is not None:
            if len(self.multivariate_probability_map) != len(self.multivariate_counts_map) or \
                    any(len(probability_row) != len(count_row)
                        for probability_row, count_row in
                        zip(self.multivariate_probability_map, self.multivariate_counts_map)):
                errors.append(
                    "The dimensions of multivariate_probability_map and multivariate_counts_map do not match.")

        # Check if the length of multivariate_support matches the columns of multivariate_probability_map
        if self.multivariate_support is not None and self.multivariate_probability_map is not None:
            if len(self.multivariate_support) != len(self.multivariate_probability_map[0]):
                errors.append(
                    "The length of multivariate_support must match the columns of multivariate_probability_map.")

        # Check if the length of multivariate_support matches the columns of multivariate_counts_map
        if self.multivariate_support is not None and self.multivariate_counts_map is not None:
            if len(self.multivariate_support) != len(self.multivariate_counts_map[0]):
                errors.append("The length of multivariate_support must match the columns of multivariate_counts_map.")

        # Return the list of errors if any, or True if no errors
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
        supports: Union[Dict, None] = None,  # Dict with: variable_name: variable_type_name
        numeric_variables_bins: int = 100,
        numeric_smoothing: bool = True,
        date_gaps_smoothing: bool = False,
        verbose: bool = False
) -> DataTemporalMap:
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

    supports: Union[Dict, None]
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
        batches without data. By default gaps are filled with NAs.

    verbose: bool
        Whether to display additional information during the process. Defaults to `False`.

    Returns
    -------
    DataTemporalMap
        The DataTemporalMap object or a dictionary of DataTemporalMap objects depending on teh number of
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

    # Get VARIABLE types, others will not be allowed
    data_types = data_without_date_column.dtypes
    float_columns = data_types == VALID_FLOAT_TYPE
    integer_columns = data_types == VALID_INTEGER_TYPE
    string_columns = data_types == VALID_STRING_TYPE
    date_columns = data_types == VALID_DATE_TYPE
    categorical_columns = data_types == VALID_CATEGORICAL_TYPE

    if verbose:
        if any(float_columns):
            print(f'Number of float columns: {sum(float_columns)}')
        if any(integer_columns):
            print(f'Number of integer columns: {sum(integer_columns)}')
        if any(string_columns):
            print(f'Number of string columns: {sum(string_columns)}')
        if any(date_columns):
            print(f'Number of date columns: {sum(date_columns)}')
        if any(categorical_columns):
            print(f'Number of categorical columns: {sum(categorical_columns)}')

    # Convert dates to numbers
    if any(date_columns):
        data_without_date_column.iloc[:, date_columns] = data_without_date_column.iloc[:, date_columns].apply(
            pd.to_numeric,
            errors='coerce'
        )
        if verbose:
            print('Converting date columns to numeric for distribution analysis')

    # Create supports
    supports_to_fill = {column: None for column in data_without_date_column.columns}
    supports_to_estimate_columns = data_without_date_column.columns.to_series()

    if supports is not None:
        for column_index, column in enumerate(supports):
            if column in supports_to_fill:
                supports_to_fill[column] = supports[column]
                supports_to_estimate_columns.drop(column)
                error_in_support = False

                if supports[column].dtypes == VALID_CATEGORICAL_TYPE:
                    error_in_support = (
                            not supports[column].dtype.name == VALID_CATEGORICAL_TYPE
                            or not supports[column].dtype.name == VALID_STRING_TYPE
                    )
                elif supports[column].dtypes == VALID_DATE_TYPE:
                    error_in_support = not supports[column].dtype.name == VALID_DATE_TYPE
                elif supports[column].dtypes == VALID_INTEGER_TYPE:
                    error_in_support = not supports[column].dtype.name == VALID_INTEGER_TYPE
                elif supports[column].dtypes == VALID_FLOAT_TYPE:
                    error_in_support = not supports[column].dtype.name == VALID_FLOAT_TYPE

                if error_in_support:
                    raise ValueError(
                        f'The provided support for variable {column} does not match with its variable type')

    supports = supports_to_fill

    if any(supports_to_estimate_columns):
        if verbose:
            print('Estimating supports from data')

        all_na = data_without_date_column.loc[:, supports_to_estimate_columns].apply(lambda x: x.isnull().all())

        # Exclude from the analysis those variables with no finite values, if any
        if any(all_na):
            if verbose:
                print(
                    f'Removing variables with no finite values: {", ".join(data_without_date_column.columns[all_na])}')
            warnings.warn(
                f'Removing variables with no finite values: {", ".join(data_without_date_column.columns[all_na])}')

            data_without_date_column = data_without_date_column.loc[:, ~all_na]
            number_of_columns = len(data_without_date_column.columns)
            supports = {column_name: data_type for column_name, data_type in supports.items() if
                        not all_na[column_name]}

            data_types = data_without_date_column.dtypes
            float_columns = data_types == VALID_FLOAT_TYPE
            integer_columns = data_types == VALID_INTEGER_TYPE
            string_columns = data_types == VALID_STRING_TYPE
            date_columns = data_types == VALID_DATE_TYPE
            categorical_columns = data_types == VALID_CATEGORICAL_TYPE

    if np.any(categorical_columns & supports_to_estimate_columns):
        data_without_date_column.loc[:,
        categorical_columns & supports_to_estimate_columns] = data_without_date_column.loc[:,
                                                              categorical_columns & supports_to_estimate_columns].apply(
            lambda col: col.cat.add_categories([MISSING_VALUE]) if col.isnull().any() else col)
        data_without_date_column.loc[:,
        categorical_columns & supports_to_estimate_columns] = data_without_date_column.loc[:,
                                                              categorical_columns & supports_to_estimate_columns].apply(
            lambda col: col.fillna(MISSING_VALUE) if col.isnull().any() else col)

        # Extract levels and assign them to supports
        selected_columns = data_without_date_column.loc[:, categorical_columns & supports_to_estimate_columns]
        levels = selected_columns.apply(lambda col: col.cat.categories)
        supports.update(
            {
                column: levels[column]
                for column
                in data_without_date_column.columns[categorical_columns & supports_to_estimate_columns]
            }
        )

    if np.any(float_columns & supports_to_estimate_columns):
        minimums = data_without_date_column.loc[:, float_columns & supports_to_estimate_columns].apply(np.nanmin,
                                                                                                       axis=0)
        maximums = data_without_date_column.loc[:, float_columns & supports_to_estimate_columns].apply(np.nanmax,
                                                                                                       axis=0)
        supports.update(
            {
                column: np.linspace(minimum, maximum, numeric_variables_bins).tolist()
                for column, minimum, maximum
                in zip(data_without_date_column.columns[float_columns & supports_to_estimate_columns], minimums,
                       maximums)
            }
        )
        if np.any(minimums == maximums):
            mask = (minimums == maximums) & float_columns & supports_to_estimate_columns
            supports.update(
                {
                    column: [value[0] for value in supports[column]]
                    for column
                    in data_without_date_column.columns[mask]
                }
            )

    if np.any(integer_columns & supports_to_estimate_columns):
        minimums = data_without_date_column.loc[:, integer_columns & supports_to_estimate_columns].apply(np.nanmin,
                                                                                                         axis=0)
        maximums = data_without_date_column.loc[:, integer_columns & supports_to_estimate_columns].apply(np.nanmax,
                                                                                                         axis=0)
        if np.sum(integer_columns & supports_to_estimate_columns) == 1:
            supports.update(
                {
                    column: np.linspace(minimum, maximum, numeric_variables_bins).tolist()
                    for column, minimum, maximum
                    in
                    zip(data_without_date_column.columns[integer_columns & supports_to_estimate_columns], minimums,
                        maximums)
                }
            )
        else:
            supports.update(
                {
                    column: np.linspace(minimum, maximum, numeric_variables_bins).tolist()
                    for column, minimum, maximum
                    in
                    zip(data_without_date_column.columns[integer_columns & supports_to_estimate_columns], minimums,
                        maximums)
                }
            )

    if np.any(string_columns & supports_to_estimate_columns):
        supports.update(
            {
                column: data_without_date_column[column].unique().tolist()
                for column
                in data_without_date_column.columns[string_columns & supports_to_estimate_columns]
            }
        )

    if np.any(date_columns & supports_to_estimate_columns):
        minimums = data_without_date_column.loc[:, date_columns & supports_to_estimate_columns].apply(np.nanmin,
                                                                                                      axis=0)
        maximums = data_without_date_column.loc[:, date_columns & supports_to_estimate_columns].apply(np.nanmax,
                                                                                                      axis=0)
        supports.update(
            {
                column: pd.date_range(minimum, maximum, periods=numeric_variables_bins).tolist()
                for column, minimum, maximum
                in zip(data_without_date_column.columns[date_columns & supports_to_estimate_columns], minimums,
                       maximums)
            }
        )

    # Convert factor variables to characters, as used by the xts Objects
    if np.any(categorical_columns):
        converted_columns = data_without_date_column.loc[:, categorical_columns].astype(
            VALID_CONVERSION_STRING_TYPE)
        data_without_date_column = data_without_date_column.assign(**converted_columns)

    # Exclude from the analysis those variables with a single value, if any
    support_lengths = [len(supports[column]) for column in data_without_date_column.columns]
    support_singles_indexes = np.array(support_lengths) < 2
    if np.any(support_singles_indexes):
        if verbose:
            print(
                f'Removing variables with less than two distinct values in their supports: {", ".join(data_without_date_column.columns[support_singles_indexes])}')
        print(
            f'The following variable/s have less than two distinct values in their supports and were excluded from the analysis: {", ".join(data_without_date_column.columns[support_singles_indexes])}')
        data_without_date_column = data_without_date_column.loc[:, ~support_singles_indexes]
        supports = {
            column: supports[column]
            for column
            in data_without_date_column.columns
        }
        data_types = data_without_date_column.dtypes
        number_of_columns = len(data_without_date_column.columns)

    if number_of_columns == 0:
        raise ValueError('Zero remaining variables to be analyzed.')

    # Estimate the Data Temporal Map
    posterior_data_classes = data_without_date_column.dtypes
    results = {}

    if verbose:
        print('Estimating the data temporal maps')

    for column_index, column in enumerate(data_without_date_column.columns, 1):
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

        if data_types[column] == VALID_DATE_TYPE:
            support = pd.DataFrame(pd.to_datetime(supports[column]))
        elif data_types[column] in [VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE]:
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
            variable_type=data_types[column],
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
        return results[data.columns[0]]


def _estimate_absolute_frequencies(data, varclass, support, numeric_smoothing=False):
    """
    Estimates the absolute frequencies of data, which will be the counts_map in the final DataTemporalMap object
    """
    data = np.array(data)
    if varclass == VALID_STRING_TYPE:
        value_counts = pd.Series(data).value_counts()
        map_data = value_counts.reindex(support, fill_value=0).values

    elif varclass == VALID_FLOAT_TYPE:
        if np.all(np.isnan(data)):
            map_data = np.array([np.nan] * len(support))
        else:
            if not numeric_smoothing:
                hist_support = np.append(support, support[-1] + (support[-1] - support[-2]))
                data = data[(data >= min(hist_support)) & (data < max(hist_support))]
                bin_edges = hist_support
                map_data, _ = np.histogram(data, bins=bin_edges)
            else:
                if np.sum(~np.isnan(data)) < 4:
                    print(
                        'Estimating a 1-dimensional kernel density smoothing with less than 4 data points can result in an inaccurate estimation.'
                        ' For more information see "Density Estimation for Statistics and Data Analysis, Bernard.W.Silverman, CRC, 1986", chapter 4.5.2 "Required sample size for given accuracy".'
                    )
                if np.sum(~np.isnan(data)) < 2:
                    data = np.repeat(data[~np.isnan(data)], 2)
                    ndata = 1
                else:
                    data = data[~np.isnan(data)]
                    ndata = np.sum(~np.isnan(data))

                kde = gaussian_kde(
                    data)
                map_data = kde(support) * ndata

    elif varclass == VALID_INTEGER_TYPE:
        if np.all(np.isnan(data)):
            map_data = np.array([np.nan] * len(support))
        else:
            hist_support = np.append(support, support[-1] + (support[-1] - support[-2]))
            data = data[(data >= min(hist_support)) & (data < max(hist_support))]
            bin_edges = hist_support
            map_data, _ = np.histogram(data, bins=bin_edges)

    else:
        raise ValueError(f'data class {varclass} not valid for distribution estimation.')

    return map_data


def estimate_multivariate_data_temporal_map(
        data: pd.DataFrame,
        date_column_name: str,
        kde_resolution: int = 10,
        dimensions: int = 2,
        period: str = TEMPORAL_PERIOD_MONTH,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        dim_reduction: str = 'PCA',
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
        A string indicating teh name of the column in data containing the analysis date variable.

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

    scatter_plot: bool
        Whether to generate a scatter plot of the first two principal components of the dimensionality reduction

    verbose: bool
        Whether to display additional information during the process. Defaults to `False`.

    Returns
    -------
    MultiVariateDataTemporalMap
        The MultivariateDataTemporalMap object of the data
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

    # Get VARIABLE types, others will not be allowed
    data_types = data_without_date_column.dtypes
    float_columns = data_types == VALID_FLOAT_TYPE
    integer_columns = data_types == VALID_INTEGER_TYPE
    string_columns = data_types == VALID_STRING_TYPE
    date_columns = data_types == VALID_DATE_TYPE
    categorical_columns = data_types == VALID_CATEGORICAL_TYPE

    if verbose:
        if any(float_columns):
            print(f'Number of float columns: {sum(float_columns)}')
        if any(integer_columns):
            print(f'Number of integer columns: {sum(integer_columns)}')
        if any(string_columns):
            print(f'Number of string columns: {sum(string_columns)}')
        if any(date_columns):
            print(f'Number of date columns: {sum(date_columns)}')
        if any(categorical_columns):
            print(f'Number of categorical columns: {sum(categorical_columns)}')

    # Convert dates to numbers
    if any(date_columns):
        data_without_date_column.iloc[:, date_columns] = data_without_date_column.iloc[:, date_columns].apply(
            pd.to_numeric,
            errors='coerce'
        )
        if verbose:
            print('Converting date columns to numeric for distribution analysis')

    if verbose:
        print(f'Applying dimensionality reduction with {dim_reduction}')

    if dim_reduction == PCA:
        method = prince.PCA(n_components=dimensions, random_state=112)
        reduced_data = method.fit_transform(data_without_date_column)
    elif dim_reduction == FAMD:
        method = prince.FAMD(n_components=dimensions, random_state=112)
        reduced_data = method.fit_transform(data_without_date_column)
    elif dim_reduction == MCA:
        method = prince.MCA(n_components=dimensions, random_state=112)
        reduced_data = method.fit_transform(data_without_date_column)

    reduced_data[[date_column_name]] = pd.DataFrame({
        date_column_name: pd.to_datetime(dates_for_batching)
    })

    if scatter_plot:
        if verbose:
            warnings.filterwarnings('ignore', category=FutureWarning)
            print(f'Plotting {dim_reduction} 2D Scatter Plot')
        fig = px.scatter(
            reduced_data.iloc[:, 0:2],
            x=0,
            y=1,
            title=f'{dim_reduction} Scatter Plot',
            template='plotly_white',
            opacity=0.7
        )

        fig.update_layout(
            title={
                'text': f'{dim_reduction} Scatter Plot',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 25}
            },
            xaxis_title={
                'text': f'PC1 ({method.eigenvalues_summary.iloc[0, 1]} variance)',
                'font': {'size': 18}
            },
            yaxis_title={
                'text': f'PC2 ({method.eigenvalues_summary.iloc[1, 1]} variance)',
                'font': {'size': 18}
            }

        )
        fig.show()
        warnings.filterwarnings('default', category=FutureWarning)

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
        scatter_plot: bool = False,
        verbose: bool = False
) -> Dict[str, MultiVariateDataTemporalMap]:
    """
    Estimates a MultivariateDataTemporalMap object for the data corresponding to each label of the DataFrame
    containing multiple variables (in columns) over time, using dimensionality reduction techniques (e.g., PCA)
    to handle high dimensional data.

    Parameters
    ----------
    data: pd.DataFrame
        A DataFrame where each row represents an individual or data point, and each column represents a
        variable. One column should represent the analysis date (typically the acquisition date).

    date_column_name: str
        A string indicating teh name of the column in data containing the analysis date variable.

    label_column_name: str
        The name of the column that contains the labels or class/category for each observation
        (used for concept shift analysis).

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

    scatter_plot: bool
        Whether to generate a scatter plot of the first two principal components of the dimensionality reduction

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

    # Get VARIABLE types, others will not be allowed
    data_types = data_without_date_column.dtypes
    float_columns = data_types == VALID_FLOAT_TYPE
    integer_columns = data_types == VALID_INTEGER_TYPE
    string_columns = data_types == VALID_STRING_TYPE
    date_columns = data_types == VALID_DATE_TYPE
    categorical_columns = data_types == VALID_CATEGORICAL_TYPE

    if verbose:
        if any(float_columns):
            print(f'Number of float columns: {sum(float_columns)}')
        if any(integer_columns):
            print(f'Number of integer columns: {sum(integer_columns)}')
        if any(string_columns):
            print(f'Number of string columns: {sum(string_columns)}')
        if any(date_columns):
            print(f'Number of date columns: {sum(date_columns)}')
        if any(categorical_columns):
            print(f'Number of categorical columns: {sum(categorical_columns)}')

    # Convert dates to numbers
    if any(date_columns):
        data_without_date_column.iloc[:, date_columns] = data_without_date_column.iloc[:, date_columns].apply(
            pd.to_numeric,
            errors='coerce'
        )
        if verbose:
            print('Converting date columns to numeric for distribution analysis')

    # Dimensionality reduction
    if verbose:
        print(f'Applying dimensionality reduction with {dim_reduction}')

    if dim_reduction == PCA:
        method = prince.PCA(n_components=dimensions, random_state=112)
        reduced_data = method.fit_transform(data_without_date_column)
    elif dim_reduction == FAMD:
        method = prince.FAMD(n_components=dimensions, random_state=112)
        reduced_data = method.fit_transform(data_without_date_column)
    elif dim_reduction == MCA:
        method = prince.MCA(n_components=dimensions, random_state=112)
        reduced_data = method.fit_transform(data_without_date_column)

    reduced_data[[label_column_name, date_column_name]] = pd.DataFrame({
        label_column_name: labels_columns,
        date_column_name: pd.to_datetime(dates_for_batching)
    })

    if scatter_plot and dimensions > 1:
        warnings.filterwarnings('ignore', category=FutureWarning)
        if verbose:
            print(f'Plotting {dim_reduction} 2D Scatter Plot divided by class')
        fig = px.scatter(
            reduced_data,
            x=0,
            y=1,
            color=label_column_name,
            title=f'{dim_reduction} Scatter Plot',
            template='plotly_white',
            opacity=0.5
        )

        fig.update_layout(
            title={
                'text': f'{dim_reduction} Scatter Plot',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 25}
            },
            xaxis_title={
                'text': f'PC1 ({method.eigenvalues_summary.iloc[0, 1]} variance)',
                'font': {'size': 18}
            },
            yaxis_title={
                'text': f'PC2 ({method.eigenvalues_summary.iloc[1, 1]} variance)',
                'font': {'size': 18}
            }

        )
        fig.show()
        warnings.filterwarnings('default', category=FutureWarning)

    reduced_data_by_label = {
        label: group.drop(columns=[label_column_name]).reset_index(drop=True)
        for label, group in reduced_data.groupby(label_column_name, observed=True)
    }

    # Generate DTMs
    concept_maps_dict = dict()
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


def _compute_kde(data_subset, xmin, xmax, kde_resolution):
    """
    Performs a Gaussian Kernel Density Estimation (KDE) on a subset of the original data to estimate
    the probability density function over a specified range, using a grid resolution determined by
    kde_resolution.
    """
    kde = gaussian_kde(data_subset.T, bw_method='silverman')  # Transpose for data compatibility
    grid = [np.linspace(start, stop, kde_resolution) for start, stop in zip(xmin, xmax)]
    mesh = np.meshgrid(*grid, indexing='ij')
    positions = np.vstack([m.ravel() for m in mesh])
    kde_values = kde(positions).reshape([kde_resolution] * len(xmin))
    return kde_values


def _normalize_kde(kde_values):
    """
    Normalizes the results of the Kernel Density Estimation (KDE) values so that the total area under the
    estimated probability density function equals 1. This ensures that the KDE represents a valid probability
    distribution.
    """
    kde_values = np.maximum(kde_values, 0)  # Set negative values to 0
    return kde_values / np.sum(kde_values)  # Normalize


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

    if dimensions == 1:
        kde1 = list()
        for date in dates_info['unique_dates']:
            if date in dates_info['value_counts'].index and dates_info['value_counts'][date] > dimensions:
                kde = _compute_kde(reduced_data[reduced_data[dates_info['date_column_name']] == date].drop(
                    columns=[dates_info['date_column_name']]),
                                   xmin[:1], xmax[:1], kde_resolution)
                kde1.append(kde)
            else:
                if verbose:
                    print(f'Not enough data for calculating {date} probability map.')
                kde = np.full((kde_resolution, kde_resolution), np.nan)
                kde1.append(kde)

        probability_map_1d = np.row_stack([_normalize_kde(kde).flatten() for kde in kde1])
        multivariate_probability_map_1d = [_normalize_kde(kde) for kde in kde1]
        multivariate_support_1d = [np.linspace(start, stop, kde_resolution) for start, stop in zip(xmin[:1], xmax[:1])]
        non_nan_mask = ~np.isnan(probability_map_1d).any(axis=1)
        non_nan_probability_map_1d = probability_map_1d[non_nan_mask]
        non_nan_counts_map_1d = np.round(non_nan_probability_map_1d * dates_info['value_counts'].values[:, np.newaxis])
        counts_map_1d = np.full(probability_map_1d.shape, np.nan)
        counts_map_1d[non_nan_mask] = non_nan_counts_map_1d
        multivariate_counts_map_1d = list()
        index = 0
        for prob_map in multivariate_probability_map_1d:
            if np.isnan(prob_map).any():
                multivariate_counts_map_1d.append(prob_map)
            else:
                multivariate_counts_map_1d.append(np.round(prob_map * dates_info['value_counts'].iloc[index]))
                index += 1

        dtm = MultiVariateDataTemporalMap(
            probability_map=probability_map_1d,
            multivariate_probability_map=multivariate_probability_map_1d,
            counts_map=counts_map_1d,
            multivariate_counts_map=multivariate_counts_map_1d,
            dates=dates_info['unique_dates'],
            support=pd.DataFrame(range(0, kde_resolution ** 2)),
            multivariate_support=multivariate_support_1d,
            variable_name='Dim.reduced.1D',
            variable_type='float64',
            period=dates_info['period']
        )

    elif dimensions == 2:
        kde2 = list()
        for date in dates_info['unique_dates']:
            if date in dates_info['value_counts'].index and dates_info['value_counts'][date] > dimensions:
                kde = _compute_kde(reduced_data[reduced_data[dates_info['date_column_name']] == date].drop(
                    columns=[dates_info['date_column_name']]),
                                   xmin[:2], xmax[:2], kde_resolution)
                kde2.append(kde)
            else:
                if verbose:
                    print(f'Not enough data for calculating {date} probability map.')
                kde = np.full((kde_resolution, kde_resolution), np.nan)
                kde2.append(kde)

        probability_map_2d = np.row_stack([_normalize_kde(kde).flatten() for kde in kde2])
        multivariate_probability_map_2d = [_normalize_kde(kde) for kde in kde2]
        multivariate_support_2d = [np.linspace(start, stop, kde_resolution) for start, stop in zip(xmin[:2], xmax[:2])]
        non_nan_mask = ~np.isnan(probability_map_2d).any(axis=1)
        non_nan_probability_map_2d = probability_map_2d[non_nan_mask]
        non_nan_counts_map_2d = np.round(non_nan_probability_map_2d * dates_info['value_counts'].values[:, np.newaxis])
        counts_map_2d = np.full(probability_map_2d.shape, np.nan)
        counts_map_2d[non_nan_mask] = non_nan_counts_map_2d
        multivariate_counts_map_2d = list()
        index = 0
        for prob_map in multivariate_probability_map_2d:
            if np.isnan(prob_map).any():
                multivariate_counts_map_2d.append(prob_map)
            else:
                multivariate_counts_map_2d.append(np.round(prob_map * dates_info['value_counts'].iloc[index]))
                index += 1

        dtm = MultiVariateDataTemporalMap(
            probability_map=probability_map_2d,
            multivariate_probability_map=multivariate_probability_map_2d,
            counts_map=counts_map_2d,
            multivariate_counts_map=multivariate_counts_map_2d,
            dates=dates_info['unique_dates'],
            support=pd.DataFrame(range(0, kde_resolution ** 2)),
            multivariate_support=multivariate_support_2d,
            variable_name='Dim.reduced.2D',
            variable_type='float64',
            period=dates_info['period']
        )
    elif dimensions == 3:
        kde3 = list()
        for date in dates_info['unique_dates']:
            if date in dates_info['value_counts'].index and dates_info['value_counts'][date] > dimensions:
                kde = _compute_kde(reduced_data[reduced_data[dates_info['date_column_name']] == date].drop(
                    columns=[dates_info['date_column_name']]),
                                   xmin, xmax, kde_resolution)
                kde3.append(kde)
            else:
                kde = np.full((kde_resolution, kde_resolution, kde_resolution), np.nan)
                kde3.append(kde)

        probability_map_3d = np.row_stack([_normalize_kde(kde).flatten() for kde in kde3])
        multivariate_probability_map_3d = [_normalize_kde(kde) for kde in kde3]
        multivariate_support_3d = [np.linspace(start, stop, kde_resolution) for start, stop in zip(xmin, xmax)]
        non_nan_mask = ~np.isnan(probability_map_3d).any(axis=1)
        non_nan_probability_map_3d = probability_map_3d[non_nan_mask]
        non_nan_counts_map_3d = np.round(non_nan_probability_map_3d * dates_info['value_counts'].values[:, np.newaxis])
        counts_map_3d = np.full(probability_map_3d.shape, np.nan)
        counts_map_3d[non_nan_mask] = non_nan_counts_map_3d
        multivariate_counts_map_3d = list()
        index = 0
        for prob_map in multivariate_probability_map_3d:
            if np.isnan(prob_map).any():
                multivariate_counts_map_3d.append(prob_map)
            else:
                multivariate_counts_map_3d.append(np.round(prob_map * dates_info['value_counts'].iloc[index]))
                index += 1

        dtm = MultiVariateDataTemporalMap(
            probability_map=probability_map_3d,
            multivariate_probability_map=multivariate_probability_map_3d,
            counts_map=counts_map_3d,
            multivariate_counts_map=multivariate_counts_map_3d,
            dates=dates_info['unique_dates'],
            support=pd.DataFrame(range(0, kde_resolution ** 3)),
            multivariate_support=multivariate_support_3d,
            variable_name='Dim.reduced.3D',
            variable_type='float64',
            period=dates_info['period']
        )
    return dtm
