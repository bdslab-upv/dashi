# Temporal periods
from enum import Enum

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
VALID_CATEGORICAL_TYPE = 'category'  # ?
# TODO: check categorical instead of R factor
VALID_TYPES_WITHOUT_DATE = [VALID_INTEGER_TYPE, VALID_STRING_TYPE, VALID_FLOAT_TYPE,
                            VALID_CATEGORICAL_TYPE]  # Pandas types
VALID_TYPES = VALID_TYPES_WITHOUT_DATE + [VALID_DATE_TYPE]  # Pandas types

# Missings
MISSING_VALUE = 'NA'

# Months
MONTH_SHORT_ABBREVIATIONS = ['J', 'F', 'M', 'A', 'm', 'j', 'x', 'a', 'S', 'O', 'N', 'D']
MONTH_LONG_ABBREVIATIONS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# Temporal map plotting sorting method
class DataTemporalMapPlotSortingMethod(Enum):
    Frequency = 'frequency'
    Alphabetical = 'alphabetical'


class DataTemporalMapPlotMode(Enum):
    Heatmap = 'heatmap'
    Series = 'series'


class PlotColorPalette(Enum):
    Spectral = 'Spectral'
    Viridis = 'Viridis'
    Viridis_reversed = 'Viridis_r'
    Magma = 'Magma'
    Magma_reversed = 'Magma_r'
