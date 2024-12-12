import io
import requests

import pandas as pd

<<<<<<< HEAD
from dashi import constants
from dashi.unsupervised_characterization.data_temporal_map.data_temporal_map import (estimate_multidim_data_temporal_map,
                                                                                     estimate_multidim_concept_shift, estimate_data_temporal_map)
from dashi.igt.igt_plotting import plot_IGT_projection
from dashi.igt.igt_projection_estimator import estimate_igt_projection
from dashi.utils import format_date
=======
from datashift import constants
from datashift.unsupervised_characterization.data_temporal_map.data_temporal_map import (estimate_multivariate_data_temporal_map,
                                                                                         estimate_conditional_data_temporal_map, estimate_univariate_data_temporal_map)
from datashift.unsupervised_characterization.data_temporal_map.data_temporal_map_plotter import plot_univariate_data_temporal_map, \
    plot_multivariate_data_temporal_map, plot_conditional_data_temporal_map
from datashift.unsupervised_characterization.igt.igt_plotting import plot_IGT_projection
from datashift.unsupervised_characterization.igt.igt_projection_estimator import estimate_igt_projection
from datashift.utils import format_date
>>>>>>> 2ff4a7787c65c95a5bb56ae21222705e2aff8e30


# EXAMPLE DATASET
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
dataset_formated['newborn'] = dataset_formated['newborn'].astype(object)
dataset_formated['race'] = dataset_formated['race'].astype(object)
dataset_formated['disstatus'] = dataset_formated['disstatus'].astype(object)
dataset_formated['dayscare'] = dataset_formated['dayscare'].astype(float)
dataset_formated['lengthflag'] = dataset_formated['lengthflag'].astype(object)
dataset_formated['region'] = dataset_formated['region'].astype(object)
dataset_formated['hospbeds'] = dataset_formated['hospbeds'].astype(float)
dataset_formated['hospownership'] = dataset_formated['hospownership'].astype(float)
dataset_formated['sex'] = dataset_formated['sex'].astype(object)

LABEL_NAME = 'sex'
data_without_label = dataset_formated.drop(columns=[LABEL_NAME])
START_DATE = pd.to_datetime('2015')
DIMENSIONS = 2
PERIOD = 'month'


# Prior probability shift
prob_maps = estimate_univariate_data_temporal_map(data=dataset_formated, date_column_name=DATE_COLUMN,
                                                  period=PERIOD, numeric_smoothing=False,
                                                  verbose=True)

plot_univariate_data_temporal_map(data_temporal_map=prob_maps[LABEL_NAME], absolute=False,
                                  sorting_method=constants.DataTemporalMapPlotSortingMethod.Frequency,
                                  color_palette=constants.PlotColorPalette.Spectral,
                                  mode=constants.DataTemporalMapPlotMode.Series)

prior_igt_projection = estimate_igt_projection(
    data_temporal_map=prob_maps[LABEL_NAME],
    dimensions=DIMENSIONS,
    embedding_type='classicalmds'
)

plot_IGT_projection(prior_igt_projection,
                    dimensions=DIMENSIONS,
                    trajectory=True)

# Covariate shift
dtm = estimate_multivariate_data_temporal_map(data=data_without_label, date_column_name=DATE_COLUMN, kde_resolution=20,
                                              dimensions=DIMENSIONS, period=PERIOD,
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

dtm_concept_shift = estimate_conditional_data_temporal_map(data=dataset_formated, date_column_name=DATE_COLUMN,
                                                           label_column_name=LABEL_NAME, kde_resolution=20,
                                                           dimensions=DIMENSIONS, period=PERIOD,
                                                           dim_reduction=constants.FAMD, scatter_plot=True,
                                                           verbose=True)

plot_conditional_data_temporal_map(data_temporal_map_dict=dtm_concept_shift)

igt_proj_concept = estimate_igt_projection(dtm_concept_shift,
                                           dimensions=DIMENSIONS,
                                           embedding_type='classicalmds')

plot_IGT_projection(igt_proj_concept,
                    dimensions=DIMENSIONS,
                    trajectory=True,
                    color_palette=constants.PlotColorPalette.Viridis)

