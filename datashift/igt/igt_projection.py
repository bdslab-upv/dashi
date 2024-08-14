from dataclasses import dataclass
from typing import Union, List

from datashift.data_temporal_map.data_temporal_map import DataTemporalMap


@dataclass
class IGTProjection:
    # TODO: change types
    data_temporal_map: Union[DataTemporalMap, None] = None
    projection: Union[List[List[float]], None] = None
    embedding_type: Union[str, None] = None
    stress: Union[float, None] = None
