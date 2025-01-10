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

# Allowed variables in the code

# Temporal periods
TEMPORAL_PERIOD_WEEK = 'week'
TEMPORAL_PERIOD_MONTH = 'month'
TEMPORAL_PERIOD_YEAR = 'year'
VALID_TEMPORAL_PERIODS = [TEMPORAL_PERIOD_WEEK, TEMPORAL_PERIOD_MONTH, TEMPORAL_PERIOD_YEAR]

# Types
VALID_DATE_TYPE = 'datetime64[ns]'
VALID_FLOAT_TYPE = 'float64'
VALID_INTEGER_TYPE = 'int64'
VALID_DEFAULT_STRING_TYPE = 'object'  # recommended string type, more efficient than object
VALID_STRING_TYPE = 'object'  # recommended string type, more efficient than object # string
VALID_CONVERSION_STRING_TYPE = 'object'  # recommended string type, more efficient than object #string
VALID_CATEGORICAL_TYPE = 'category'
VALID_TYPES_WITHOUT_DATE = [VALID_INTEGER_TYPE, VALID_STRING_TYPE, VALID_FLOAT_TYPE,
                            VALID_CATEGORICAL_TYPE]  # Pandas types
VALID_TYPES = VALID_TYPES_WITHOUT_DATE + [VALID_DATE_TYPE]  # Pandas types

# Missings
MISSING_VALUE = 'NA'

# Months
MONTH_SHORT_ABBREVIATIONS = ['J', 'F', 'M', 'A', 'm', 'j', 'x', 'a', 'S', 'O', 'N', 'D']
MONTH_LONG_ABBREVIATIONS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Dimensionality reduction
FAMD = 'FAMD'
PCA = 'PCA'
MCA = 'MCA'
VALID_DIM_REDUCTION_TYPES = [FAMD, PCA, MCA]

# Sorting method
Frequency = 'frequency'
Alphabetical = 'alphabetical'
VALID_SORTING_METHODS = [Frequency, Alphabetical]

# Plot mode
Heatmap = 'heatmap'
Series = 'series'
VALID_PLOT_MODES = [Heatmap, Series]

# Color scale
Spectral = 'Spectral'
Viridis = 'viridis'
Viridis_reversed = 'viridis_r'
Magma = 'magma'
Magma_reversed = 'magma_r'
VALID_COLOR_PALETTES = [Spectral, Viridis,Viridis_reversed, Magma, Magma_reversed]
