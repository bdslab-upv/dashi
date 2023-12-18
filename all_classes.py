from typing import Union, List
from dataclasses import dataclass

from all_constants import VALID_TEMPORAL_PERIODS, VALID_TYPES


@dataclass
class DataTemporalMap:
    # TODO: change types
    # Example
    # probability_map = [
    #   [ 0, 1, 2...],
    #   [ 0, 1, 2...],
    #   [ 0, 1, 2...]
    # ]
    probability_map: Union[List[List[float]], None] = None
    counts_map: Union[List[List[int]], None] = None
    dates: Union[List[str], None] = None
    support: Union[List[str], None] = None
    variable_name: Union[str, None] = None
    variable_type: Union[str, None] = None
    period: Union[str, None] = None


@dataclass
class IGTProjection:
    # TODO: change types
    data_temporal_map: Union[DataTemporalMap, None] = None
    projection: Union[List[List[float]], None] = None
    embedding_type: Union[str, None] = None
    stress: Union[float, None] = None


def check_data_temporal_map(data_temporal_map: DataTemporalMap) -> Union[List[str], bool]:
    errors = []

    # TODO: Quizás hacer log de los errores en vez de devolver varios tipos y sacar solo boolean

    # Check if the dimensions of probability_map and counts_map match
    if data_temporal_map.probability_map is not None and data_temporal_map.counts_map is not None:
        if (len(data_temporal_map.probability_map) != len(data_temporal_map.counts_map)
                or any(len(probability_row) != len(count_row) for probability_row, count_row in
                       zip(data_temporal_map.probability_map, data_temporal_map.counts_map))):
            errors.append("the dimensions of probability_map and counts_map do not match")

    # Check if the length of dates matches the rows of probability_map
    if data_temporal_map.dates is not None and data_temporal_map.probability_map is not None:
        if len(data_temporal_map.dates) != len(data_temporal_map.probability_map):
            errors.append("the length of dates must match the rows of probability_map")

    # Check if the length of dates matches the rows of counts_map
    if data_temporal_map.dates is not None and data_temporal_map.counts_map is not None:
        if len(data_temporal_map.dates) != len(data_temporal_map.counts_map):
            errors.append("the length of dates must match the rows of counts_map")

    # Check if the length of support matches the columns of probability_map
    if data_temporal_map.support is not None and data_temporal_map.probability_map is not None:
        if len(data_temporal_map.support) != len(data_temporal_map.probability_map):
            errors.append("the length of support must match the columns of probability_map")

    # Check if the length of support matches the columns of counts_map
    if data_temporal_map.support is not None and data_temporal_map.counts_map is not None:
        if len(data_temporal_map.support) != len(data_temporal_map.counts_map):
            errors.append("the length of support must match the columns of counts_map")

    # Check if period is one of the valid periods
    if data_temporal_map.period is not None and data_temporal_map.period not in VALID_TEMPORAL_PERIODS:
        errors.append(f"period must be one of the following: {', '.join(VALID_TEMPORAL_PERIODS)}")

    # Check if variableType is one of the valid types
    # TODO: duda, variable type es de uno?
    if data_temporal_map.variable_type is not None and data_temporal_map.variable_type not in VALID_TYPES:
        errors.append(f"variable_type must be one of the following: {', '.join(VALID_TYPES)}")

    return errors if errors else True
