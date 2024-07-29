import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Union, List, Dict

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from constants import VALID_TEMPORAL_PERIODS, VALID_TYPES, VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE, \
    VALID_INTEGER_TYPE, VALID_FLOAT_TYPE, \
    VALID_DATE_TYPE, TEMPORAL_PERIOD_WEEK, TEMPORAL_PERIOD_MONTH, TEMPORAL_PERIOD_YEAR, VALID_CONVERSION_STRING_TYPE, \
    MISSING_VALUE, VALID_TYPES_WITHOUT_DATE


@dataclass
class DataTemporalMap:
    # TODO: change types
    # TODO: hablar con Ángel la definición de los atributos
    # Example
    # probability_map = [
    #   [ 0, 1, 2...],
    #   [ 0, 1, 2...],
    #   [ 0, 1, 2...]
    # ]
    probability_map: Union[List[List[float]], None] = None
    counts_map: Union[List[List[int]], None] = None
    dates: Union[List[datetime], None] = None
    support: Union[List[str], None] = None
    variable_name: Union[str, None] = None
    variable_type: Union[str, None] = None
    period: Union[str, None] = None

    def check(self) -> Union[List[str], bool]:
        errors = []

        # TODO: Quizás hacer log de los errores en vez de devolver varios tipos y sacar solo boolean

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
        # TODO: duda, variable type es de uno?
        if self.variable_type is not None and self.variable_type not in VALID_TYPES:
            errors.append(f"variable_type must be one of the following: {', '.join(VALID_TYPES)}")

        return errors if errors else True


def trim_data_temporal_map(
        data_temporal_map: DataTemporalMap,
        start_date: datetime = None,
        end_date: datetime = None
) -> DataTemporalMap:
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


def estimate_data_temporal_map(
        data: pd.DataFrame,
        date_column_name,
        period=TEMPORAL_PERIOD_MONTH,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        supports: Union[Dict, None] = None,  # Dict with: variable_name: variable_type_name
        numeric_variables_bins=100,
        numeric_smoothing=True,
        date_gaps_smoothing=False,
        verbose=False
):
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

    # TODO: check classes
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

    # Get variable types, others will not be allowed
    data_types = data_without_date_column.dtypes
    float_columns = data_types == VALID_FLOAT_TYPE
    integer_columns = data_types == VALID_INTEGER_TYPE
    string_columns = data_types == VALID_STRING_TYPE
    date_columns = data_types == VALID_DATE_TYPE
    categorical_columns = data_types == VALID_CATEGORICAL_TYPE  # TODO: check, categorical, pandas

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
    supports_to_estimate_columns = data_without_date_column.columns.to_list()

    # TODO: to test
    if supports is not None:
        for column_index, column in enumerate(supports):
            if column in supports_to_fill:
                supports_to_fill[column] = supports[column]
                supports_to_estimate_columns.remove(column)
                error_in_support = False

                # TODO: Check this (category)
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

    # TODO: To test, define category of NA/None
    if np.any(categorical_columns & supports_to_estimate_columns):
        # Crear categoría para los NA, missings
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

    # Tested
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

    # DUDA if
    # Tested
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

    # TESTED
    if np.any(string_columns & supports_to_estimate_columns):
        supports.update(
            {
                column: data_without_date_column[column].unique().tolist()
                for column
                in data_without_date_column.columns[string_columns & supports_to_estimate_columns]
            }
        )

    # TODO: To test
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

        if start_date is not None or end_date is not None:
            if start_date is None:
                start_date = min(dates)
            if end_date is None:
                end_date = max(dates)

            data_xts = data_xts[start_date:end_date]

        period_function = {
            TEMPORAL_PERIOD_WEEK: data_xts.resample('W').apply(
                estimate_absolute_frequencies,
                varclass=posterior_data_classes[column],
                support=supports[column],
                numeric_smoothing=numeric_smoothing
            ),
            TEMPORAL_PERIOD_MONTH: data_xts.resample('MS').apply(
                estimate_absolute_frequencies,
                varclass=posterior_data_classes[column],
                support=supports[column],
                numeric_smoothing=numeric_smoothing
            ),
            TEMPORAL_PERIOD_YEAR: data_xts.resample('YS').apply(
                estimate_absolute_frequencies,
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


def estimate_absolute_frequencies(data, varclass, support, numeric_smoothing=False):
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
                    data)  # combina muchas gaussianas, una por punto, kdtree, revisar criterio de bandwith, semilla aleatoria, silverman
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

def estimate_multidim_data_temporal_map(
        data: pd.DataFrame,
        dates: datetime.date,
        binsize=10,
        period=TEMPORAL_PERIOD_MONTH,
):
    if data is None:
        raise ValueError('An input data frame is required.')

    if period not in VALID_TEMPORAL_PERIODS:
        raise ValueError(f'Period must be one of the following: {", ".join(VALID_TEMPORAL_PERIODS)}')


    xmin = data.min(axis=0)
    xmax = data.max(axis=0)

    if period == TEMPORAL_PERIOD_MONTH:
        dates_for_batching = pd.to_datetime(dates).to_period('M')
        unique_dates = sorted(dates_for_batching.unique())
    elif period == TEMPORAL_PERIOD_YEAR:
        dates_for_batching = pd.to_datetime(dates).dt.year
        unique_dates = dates_for_batching.unique()
        unique_dates = pd.Series([f'{year}-01-01' for year in unique_dates])
    elif period == TEMPORAL_PERIOD_WEEK:
        dates_for_batching = pd.to_datetime(dates).to_period('W')
        unique_dates = sorted(dates_for_batching.unique())

    kde2 = data.groupby(dates_for_batching).apply(lambda group: compute_kde(data.loc[group.index, [0, 1]], xmin[:2], xmax[:2], binsize)).tolist()
    probability_map_2d = np.column_stack([process_kde(kde).flatten() for kde in kde2])
    probability_map_2d = pd.DataFrame(data=probability_map_2d, columns=unique_dates)


    kde3 = data.groupby(dates_for_batching).apply(lambda group: compute_kde(data.loc[group.index, 0:4], xmin, xmax, binsize)).tolist()

    print(kde3)

def compute_kde(data_subset, xmin, xmax, binsize):
    kde = gaussian_kde(data_subset.T, bw_method='silverman')
    grid = [np.linspace(start, stop, binsize) for start, stop in zip(xmin, xmax)]
    mesh = np.meshgrid(*grid)
    positions = np.vstack([m.ravel() for m in mesh])
    kde_values = kde(positions).reshape([binsize] * len(xmin))

    return kde_values

def process_kde(kde_values):
    kde_values = np.maximum(kde_values, 0) #Set negative values to 0
    return kde_values / np.sum(kde_values)

