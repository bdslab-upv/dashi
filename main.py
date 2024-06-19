import io

import pandas as pd
import requests

import constants
from data_temporal_map.data_temporal_map import estimate_data_temporal_map
from data_temporal_map.data_temporal_map_plotter import plot_data_temporal_map
from igt.igt_plotting import plot_IGT_projection
from igt.igt_projection_estimator import estimate_igt_projection
from utils import format_date

# URL to the dataset
url = 'http://github.com/hms-dbmi/EHRtemporalVariability-DataExamples/raw/master/nhdsSubset.csv'
downloaded_dataset = requests.get(url).content
dataset_string = io.StringIO(downloaded_dataset.decode('utf-8'))

# Read the CSV file
dataframe = pd.read_csv(dataset_string, sep=',', na_values='')

# TODO: pre conversion
# converted_columns = dataframe.loc[:, dataframe.dtypes == VALID_DEFAULT_STRING_TYPE].astype('string')
# dataframe = dataframe.assign(**converted_columns)

# TO TEST
dataframe["age"] = dataframe["age"].astype('float')
dataframe['sex'] = dataframe.sex.astype('category')

# Display the first few rows of the dataset
pd.set_option('display.max_columns', None)

# Formatted dataset with dates
dataset_formatted = format_date(dataframe, 'date', date_format='%y/%m', verbose=True)

# Testing: counts
#    - string: done (diagcode1)
#    - int: done (age as integer)
#    - categorical: (sex as category)
#    - float: done (age as float) (numericSmoothing = False nice y True da poco distinto por kde), comprobar que se normaliza.
# Testing: frequencies
#    - string: done (diagcode1)
#    - int: done  (age as integer)
#    - categorical: done (sex as category)
#    - float: done (age as float) (numericSmoothing = False nice y True da poco distinto por kde), comprobar que se normaliza.


prob_maps = estimate_data_temporal_map(
    data=dataset_formatted,
    date_column_name='date',
    period=constants.TEMPORAL_PERIOD_MONTH,
    verbose=True,
    numeric_smoothing=False
)

plot_data_temporal_map(
    data_temporal_map=prob_maps['diagcode1'],
    start_value=0,
    end_value=20,
    # start_date=min(prob_maps['diagcode1'].dates),
    # end_date=max(prob_maps['diagcode1'].dates),
    # start_date=datetime.datetime(year=2000, month=1, day=2),
    # end_date=datetime.datetime(year=2005, month=12, day=31),
    color_palette=constants.PlotColorPalette.Spectral,
    absolute=False,
    sorting_method=constants.DataTemporalMapPlotSortingMethod.Frequency,
    mode=constants.DataTemporalMapPlotMode.Heatmap,
    plot_title='BULERIA BULERIA, MÁS TE QUIERO CADA DÍA'
)

igt_projection = estimate_igt_projection(
    data_temporal_map=prob_maps['diagcode1'],
    dimensions=2,
    # start_date=datetime.datetime(year=2000, month=1, day=2),
    # end_date=datetime.datetime(year=2005, month=12, day=31),
    # embedding_type='nonmetricmds'
)

# # TODO test dimensions with dates, colors and trajectory
# # TODO to check, in 3d red up blue down, David
plot_IGT_projection(
    igt_projection=igt_projection,
    dimensions=2,
    trajectory=True
)

# TODO Los phw esos
# TODO Mirar colorinchis

# TODO Condiciconadas
# TODO Pasar datos y modelos en un periodo
