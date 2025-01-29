import io

import pandas as pd
import requests

from dashi.unsupervised_characterization.data_temporal_map.data_temporal_map import \
    estimate_univariate_data_temporal_map
from dashi.unsupervised_characterization.data_temporal_map.data_temporal_map_plotter import \
    plot_univariate_data_temporal_map
from dashi.unsupervised_characterization.igt.igt_plotting import plot_IGT_projection
from dashi.unsupervised_characterization.igt.igt_projection_estimator import estimate_igt_projection
from dashi.utils import format_data

# URL to the dataset
url = 'http://github.com/hms-dbmi/EHRtemporalVariability-DataExamples/raw/master/nhdsSubset.csv'
downloaded_dataset = requests.get(url).content
dataset_string = io.StringIO(downloaded_dataset.decode('utf-8'))

# Read the CSV file
dataframe = pd.read_csv(dataset_string, sep=',', na_values='')

# converted_columns = dataframe.loc[:, dataframe.dtypes == VALID_DEFAULT_STRING_TYPE].astype('string')
# dataframe = dataframe.assign(**converted_columns)

# TO TEST
dataframe["age"] = dataframe["age"].astype('float64')
dataframe['sex'] = dataframe.sex.astype('object')
dataframe['diagcode1'] = dataframe.diagcode1.astype('object')
VARIABLE = 'diagcode1'

# Display the first few rows of the dataset
pd.set_option('display.max_columns', None)

# Formatted dataset with dates
dataset_formatted = format_data(dataframe, date_format='%y/%m', verbose=True)


prob_maps = estimate_univariate_data_temporal_map(data=dataset_formatted, date_column_name='date',
                                                  period=constants.TEMPORAL_PERIOD_YEAR, numeric_smoothing=False,
                                                  verbose=True)

plot_univariate_data_temporal_map(data_temporal_map=prob_maps[VARIABLE], absolute=False, log_transform=False,
                                  sorting_method=constants.DataTemporalMapPlotSortingMethod.Frequency,
                                  color_palette=constants.PlotColorPalette.Spectral,
                                  mode=constants.DataTemporalMapPlotMode.Heatmap)

igt_projection = estimate_igt_projection(
    data_temporal_map=prob_maps[VARIABLE],
    dimensions=3,
    embedding_type='pca'
)

plot_IGT_projection(
    igt_projection=igt_projection,
    dimensions=3,
    trajectory=False
)
