from datetime import datetime
from typing import Optional, List

import pandas as pd

from dashi.constants import MONTH_SHORT_ABBREVIATIONS, VALID_DATE_TYPE


def _format_date_for_year(date: datetime) -> str:
    year_part = date.strftime('%Y')

    return year_part


def _format_date_for_month(date: datetime) -> str:
    year_part = date.strftime('%y')
    month_part = MONTH_SHORT_ABBREVIATIONS[date.month - 1]

    return year_part + month_part


def _format_date_for_week(date: datetime) -> str:
    year_part = date.strftime('%y')
    month_part = MONTH_SHORT_ABBREVIATIONS[date.month - 1]
    day_part = str(date.isoweekday())

    return year_part + month_part + day_part


def format_data(input_dataframe: pd.DataFrame,
                *,
                date_column_name: str = None,
                date_format: str = '%y/%m/%d',
                verbose: bool = False,
                inputs_numerical_column_names: Optional[List[str]] = None,
                inputs_categorical_column_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Function to transform dates into 'Date' Python format

    Parameters
    ----------
    input_dataframe : pd.DataFrame
        Pandas dataframe object with at least one columns of dates.

    date_column_name: str
        The name of the column containing the dates.

    date_format: str
        Structure of date format. By default '%y/%m/%d'.

    verbose: bool
        Whether to display additional information during the process. Defaults to `False`.

    inputs_numerical_column_names: Optional[List[str]]
        A list of numerical column names in the dataset. If this parameter is `None`, the variables types must be
        managed by the user.

    inputs_categorical_column_names: Optional[List[str]]
        A list of categorical column names in the dataset. If this parameter is `None`, the variables types must be
        managed by the user.

    Returns
    -------
    pd.DataFrame
        An object of class pd.DataFrame with the date columns transformed into 'Date' Python format.
    """
    if date_column_name not in input_dataframe.columns:
        raise ValueError(f'There is no column in your DataFrame named as: {date_column_name}')

    output_dataframe = input_dataframe.copy()

    if inputs_numerical_column_names is not None:
        for col in inputs_numerical_column_names:
            output_dataframe[col] = output_dataframe[col].astype(float)

    if inputs_categorical_column_names is not None:
        for col in inputs_categorical_column_names:
            output_dataframe[col] = output_dataframe[col].astype('category')

    if output_dataframe[date_column_name].dtype == VALID_DATE_TYPE:
        return output_dataframe

    if verbose:
        print(f'Formatting the {date_column_name} column')

    if (_is_letter_in_date_format(date_format, ['Y', 'y'])
            and _is_letter_in_date_format(date_format, ['m', 'M', 'b', 'B', 'h'])
            and _is_letter_in_date_format(date_format, ['d', 'D'])):
        if verbose:
            print('The data format contains year, month, and day')
    elif (_is_letter_in_date_format(date_format, ['Y', 'y'])
          and _is_letter_in_date_format(date_format, ['m', 'M', 'b', 'B', 'h'])):
        if verbose:
            print('The data format contains year and month but not day')
            print(
                'Take into account that if you perform an analysis by week, the day will be automatically assigned as the first day of the month.')
    elif _is_letter_in_date_format(date_format, ['Y', 'y']):
        if verbose:
            print('The data format contains only the year')
            print(
                'Take into account that if you perform an analysis by week or by month, they will be automatically assigned as the first day of the month and first month of the year.')
    else:
        print('Please, check the format of the date. At least it should contain the year.')
        raise ValueError('Invalid date format')

    output_dataframe[date_column_name] = pd.to_datetime(output_dataframe[date_column_name], format=date_format)

    # Check if there are rows with na
    # If there are rows with na remove the complete rows
    date_rows_without_na = output_dataframe.dropna(subset=[date_column_name])
    if len(date_rows_without_na) == len(output_dataframe):
        return output_dataframe
    else:
        output_dataframe = date_rows_without_na.copy()
        print(
            f'There are {len(output_dataframe) - len(date_rows_without_na)} rows that do not contain date information. They have been removed.')
        return output_dataframe


def _is_letter_in_date_format(date_format, date_pattern):
    # Check if any of the pattern elements is on the date_format string
    return any(any(element in character_to_check for element in date_pattern) for character_to_check in date_format)
