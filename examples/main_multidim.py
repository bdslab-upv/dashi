import io
import requests

import pandas as pd

from dashi import constants
from dashi.data_temporal_map.data_temporal_map import (estimate_multidim_data_temporal_map,
                                                       estimate_multidim_concept_shift, estimate_data_temporal_map)
from dashi.data_temporal_map.data_temporal_map_plotter import plot_data_temporal_map, \
    plot_multivariate_data_temporal_map, plot_multivariate_concept_shift
from dashi.igt.igt_plotting import plot_IGT_projection
from dashi.igt.igt_projection_estimator import estimate_igt_projection
from dashi.utils import format_date

"""
# EXAMPLE DATASET R
# URL to the dataset
url = 'http://github.com/hms-dbmi/EHRtemporalVariability-DataExamples/raw/master/nhdsSubset.csv'
downloaded_dataset = requests.get(url).content
dataset_string = io.StringIO(downloaded_dataset.decode('utf-8'))

# Read the CSV file
dataset = pd.read_csv(dataset_string, sep=',', na_values='')
dataset = dataset.iloc[:, 0:12]

DATE_COLUMN = 'date'
dataset_formated = format_date(dataset, DATE_COLUMN, '%y/%m')
dataset_formated['age'] = dataset_formated['age'].astype(float)
dataset_formated['dayscare'] = dataset_formated['dayscare'].astype(float)
dataset_formated['sex'] = dataset_formated['sex'].astype(object)

LABEL_NAME = 'sex'
data_pca = dataset_formated.drop(columns=[LABEL_NAME])
START_DATE = pd.to_datetime('2015')
DIMENSIONS = 3
"""

# EXAMPLE DATASET SIMULATED
path = r'C:\Users\David\Desktop\Datasets\Simulated_Carlos\uci_heart_disease_simulshift_tsne.csv'
dataset = pd.read_csv(path, sep=';')
describe = dataset.describe()
print(dataset.dtypes)

DATE_COLUMN = 'synthetic_date'
LABEL_NAME = 'class_label'
DIMENSIONS = 3

dataset_formated = format_date(dataset, DATE_COLUMN, '%Y-%m-%d')
dataset_formated['age'] = dataset_formated['age'].astype(float)
dataset_formated['trestbps'] = dataset_formated['trestbps'].astype(float)
dataset_formated['chol'] = dataset_formated['chol'].astype(float)
dataset_formated['thalach'] = dataset_formated['thalach'].astype(float)
dataset_formated['oldpeak'] = dataset_formated['oldpeak'].apply(lambda x: float(x.replace(',', '.')))
dataset_formated['ca'] = dataset_formated['ca'].astype(float)
dataset_formated[LABEL_NAME] = dataset_formated[LABEL_NAME].astype(object)

data_pca = dataset_formated.drop(columns=[LABEL_NAME])


# Prior probability shift
prob_maps = estimate_data_temporal_map(
    data=dataset_formated,
    date_column_name=DATE_COLUMN,
    period=constants.TEMPORAL_PERIOD_YEAR,
    numeric_smoothing=False,
    verbose=True
)

plot_data_temporal_map(
    data_temporal_map=prob_maps[LABEL_NAME],
    color_palette=constants.PlotColorPalette.Spectral,
    absolute=False,
    sorting_method=constants.DataTemporalMapPlotSortingMethod.Frequency,
    mode=constants.DataTemporalMapPlotMode.Series
)

prior_igt_projection = estimate_igt_projection(
    data_temporal_map=prob_maps[LABEL_NAME],
    dimensions=DIMENSIONS,
    embedding_type='classicalmds'
)

plot_IGT_projection(prior_igt_projection,
                    dimensions=DIMENSIONS,
                    trajectory=True)

# Covariate shift
dtm = estimate_multidim_data_temporal_map(data=data_pca, date_column_name=DATE_COLUMN, kde_resolution=30,
                                          dimensions=DIMENSIONS, period=constants.TEMPORAL_PERIOD_YEAR,
                                          dim_reduction=constants.FAMD, scatter_plot=False, verbose=True)

plot_multivariate_data_temporal_map(data_temporal_map=dtm)

igt_proj_covariate = estimate_igt_projection(dtm,
                                             dimensions=DIMENSIONS,
                                             embedding_type='classicalmds')

plot_IGT_projection(igt_proj_covariate,
                    dimensions=DIMENSIONS,
                    trajectory=True,
                    color_palette=constants.PlotColorPalette.Viridis)

# Concept shift

dtm_concept_shift = estimate_multidim_concept_shift(data=dataset_formated, date_column_name=DATE_COLUMN,
                                                    label_column_name=LABEL_NAME, kde_resolution=30,
                                                    dimensions=DIMENSIONS, period=constants.TEMPORAL_PERIOD_YEAR,
                                                    dim_reduction=constants.FAMD, scatter_plot=True, verbose=True)

plot_multivariate_concept_shift(data_temporal_map_dict=dtm_concept_shift)

igt_proj_concept = estimate_igt_projection(dtm_concept_shift,
                                           dimensions=DIMENSIONS,
                                           embedding_type='classicalmds')

plot_IGT_projection(igt_proj_concept,
                    dimensions=DIMENSIONS,
                    trajectory=True,
                    color_palette=constants.PlotColorPalette.Viridis)

