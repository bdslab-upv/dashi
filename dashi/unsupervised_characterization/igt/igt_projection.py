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
Information Geometric Temporal class creation
"""

from dataclasses import dataclass
from typing import Union, List

from dashi.unsupervised_characterization.data_temporal_map.data_temporal_map import DataTemporalMap, MultiVariateDataTemporalMap


@dataclass
class IGTProjection:
    """
    Class IGTProjection object contains the estimated Information Geometric Temporal projection
    of a DataTemporalMap or MultiVariateDataTemporalMap objects

    Attributes
    ----------
    data_temporal_map : Union[DataTemporalMap, MultiVariateDataTemporalMap, None]
        A `DataTemporalMap` or 'MultiVariateDataTemporalMap' object containing the temporal data map
        that was used for generating the projection.

    projection : Union[List[List[float]], None]
        A 2D matrix matrix of floats representing the lower-dimensional projection of the temporal data.
        Each row corresponds to a data timestamp, with each column representing a dimension of the projection.

    embedding_type : Union[str, None]
        A string representing the type of embedding used for the projection (e.g., "classicalmds", "nonmetricmds",
        "pca").

    stress : Union[float, None]
        A float value representing the stress (or error) of the projection if it is available (used in MDS
        or similar techniques). The lower the stress value, the better the projection reflects the original data.
        This attribute is `None` if stress is not computed or available.
    """
    data_temporal_map: Union[DataTemporalMap, MultiVariateDataTemporalMap, None] = None
    projection: Union[List[List[float]], None] = None
    embedding_type: Union[str, None] = None
    stress: Union[float, None] = None

