import pandas as pd

from all_constants import VALID_DATE_TYPE


def format_date(input_dataframe, date_column, date_format='%y/%m/%d', verbose=False):
    if date_column not in input_dataframe.columns:
        raise ValueError(f'There is no column in your DataFrame named as: {date_column}')

    output_dataframe = input_dataframe.copy()

    if output_dataframe[date_column].dtype == VALID_DATE_TYPE:
        return output_dataframe

    if verbose:
        print(f'Formatting the {date_column} column')

    if (__is_letter_in_date_format(date_format, ['Y', 'y'])
            and __is_letter_in_date_format(date_format, ['m', 'M', 'b', 'B', 'h'])
            and __is_letter_in_date_format(date_format, ['d', 'D'])):
        if verbose:
            print('The data format contains year, month, and day')
    elif (__is_letter_in_date_format(date_format, ['Y', 'y'])
          and __is_letter_in_date_format(date_format, ['m', 'M', 'b', 'B', 'h'])):
        if verbose:
            print('The data format contains year and month but not day')
            print('Take into account that if you perform an analysis by week, the day will be automatically assigned as the first day of the month.')
    elif __is_letter_in_date_format(date_format, ['Y', 'y']):
        if verbose:
            print('The data format contains only the year')
            print('Take into account that if you perform an analysis by week or by month, they will be automatically assigned as the first day of the month and first month of the year.')
    else:
        print('Please, check the format of the date. At least it should contain the year.')
        raise ValueError('Invalid date format')

    output_dataframe[date_column] = pd.to_datetime(output_dataframe[date_column], format=date_format)

    # Check if there are rows with na
    # If there are rows with na remove the complete rows
    date_rows_without_na = output_dataframe.dropna(subset=[date_column])
    if len(date_rows_without_na) == len(output_dataframe):
        return output_dataframe
    else:
        output_dataframe = date_rows_without_na.copy()
        print(f'There are {len(output_dataframe) - len(date_rows_without_na)} rows that do not contain date information. They have been removed.')
        return output_dataframe


def __is_letter_in_date_format(date_format, date_pattern):
    # Check if any of the pattern elements is on the date_format string
    return any(any(element in character_to_check for element in date_pattern) for character_to_check in date_format)
