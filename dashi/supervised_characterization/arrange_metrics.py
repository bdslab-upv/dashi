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
Functions for arranging metrics in a data frame from the original metrics dictionary.
"""

from typing import Dict, Union, Tuple

from pandas import Series, DataFrame


def arrange_performance_metrics(*, metrics: Dict[str, float], metric_name: str) -> DataFrame:
    """
    Organizes and formats a subset of metrics from a dictionary into a pandas DataFrame.

    This function filters the metrics based on the provided metric name, selects only those
    relevant to the 'test' subset, and formats the result into a DataFrame with a corrected
    index.

    Parameters
    ----------
    metrics : dict of {str: float}
        A dictionary containing the calculated metrics for each batch and model combination.

    metric_name : str
        The name of the metric to be selected from the `metrics` dictionary.
        Regression metric names, when applicable:
            - 'MEAN_ABSOLUTE_ERROR'
            - 'MEAN_SQUARED_ERROR'
            - 'ROOT_MEAN_SQUARED_ERROR'
            - 'R_SQUARED'
        Classification metric names, when applicable:
            - 'AUC_{class_identifier}'
            - 'AUC_MACRO'
            - 'LOGLOSS'
            - 'RECALL_{class_identifier}'
            - 'PRECISION_{class_identifier}'
            - 'F1-SCORE_{class_identifier}'
            - 'ACCURACY'
            - 'RECALL_MACRO'
            - 'RECALL_MICRO'
            - 'RECALL_WEIGHTED'
            - 'PRECISION_MACRO'
            - 'PRECISION_MICRO'
            - 'PRECISION_WEIGHTED'
            - 'F1-SCORE_MACRO'
            - 'F1-SCORE_MICRO'
            - 'F1-SCORE_WEIGHTED'

    Returns
    -------
    pandas.DataFrame
        A DataFrame where the rows represent the combinations and the columns represent
        the metric values, with the index corrected for cumulative learning strategies.

    Raises
    ------
    TypeError
        If `metrics` is not a dictionary or if `metric_name` is not a string.

    Notes
    -----
    The function assumes that the `metrics` dictionary contains a third element 'test' for
    filtering the relevant metrics.
    """

    # Inputs checking
    if type(metrics) is not dict:
        raise TypeError('Metrics should be specified in a dictionary.')
    if type(metric_name) is not str:
        raise TypeError('Metric identifier needs to be specified as a string.')

    # Selection of the metrics relative to the test set
    metrics_test = {(combination[0], combination[1]): metrics_[metric_name]
                    for combination, metrics_ in metrics.items() if combination[2] == 'test'}

    # Data formatting
    metrics_test_frame = Series(metrics_test).unstack(sort=False)

    # Metrics test frame formatting
    metrics_test_frame.index = metrics_test_frame.index.map(_correct_index)

    # Output
    return metrics_test_frame


def _correct_index(index_value: Union[str, float, Tuple]) -> Union[str, float]:
    """
    Corrects the index values when learning was based on a cumulative learning strategy.

    This function handles cases where the index value is a tuple (indicating a range)
    and formats it into a string. If the index is not a tuple, the value is returned
    without modification.

    Parameters
    ----------
    index_value : str, float, or tuple
        The value to be corrected. If it is a tuple, it will be formatted as 'From X to Y',
        where X and Y are the first and last elements of the tuple, respectively.

    Returns
    -------
    str or float
        The corrected index value. If the input is a tuple, returns a string in the format
        'From X to Y'. Otherwise, returns the value unchanged.
    """

    # Correction for cumulative learning strategy
    if type(index_value) is tuple:
        index_first = index_value[0]
        index_last = index_value[-1]

        return f'From {index_first} to {index_last}'

    # Pass without modification
    else:
        return index_value
