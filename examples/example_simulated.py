import pandas as pd

from dashi import constants
from dashi.unsupervised_characterization.data_temporal_map.data_temporal_map import (
    estimate_multivariate_data_temporal_map,
    estimate_conditional_data_temporal_map, estimate_univariate_data_temporal_map)
from dashi.unsupervised_characterization.data_temporal_map.data_temporal_map_plotter import \
    plot_univariate_data_temporal_map, \
    plot_multivariate_data_temporal_map, plot_conditional_data_temporal_map
from dashi.unsupervised_characterization.igt.igt_plotting import plot_IGT_projection
from dashi.unsupervised_characterization.igt.igt_projection_estimator import estimate_igt_projection
from dashi.utils import format_data

from dashi.supervised_characterization.estimate_models import estimate_multibatch_models
from dashi.supervised_characterization.plot_performance import plot_multibatch_performance

# EXAMPLE DATASET
path = r'C:\Users\David\Desktop\Datasets\Simulated_Carlos\uci_heart_disease_full_simulshift_pcamix_cs_dim1_2y_nout.csv'
# Read the CSV file
dataset = pd.read_csv(path, sep=';')

DATE_COLUMN = 'synthetic_date'
CATEGORICAL_VARIABLES = ['sex', 'cp', 'fbs', 'restecg', 'slope', 'thal', 'ca']
NUMERICAL_VARIABLES = ['age', 'trestbps', 'chol', 'thalach']
dataset_formated = format_data(dataset, date_column_name=DATE_COLUMN, date_format='%Y-%m-%d',
                               inputs_numerical_column_names=NUMERICAL_VARIABLES,
                               inputs_categorical_column_names=CATEGORICAL_VARIABLES)
dataset_formated['oldpeak'] = pd.Series(map(lambda x: float(x.replace(',', '.')), dataset_formated['oldpeak']))

LABEL_NAME = 'class_label'
data_without_label = dataset_formated.drop(columns=[LABEL_NAME])
DIMENSIONS = 2
PERIOD = 'month'
REDUCTION_METHOD = 'FAMD'

# UNSUPERVISED

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
                    trajectory=True,
                    color_palette=constants.PlotColorPalette.Spectral)

# Covariate shift
dtm = estimate_multivariate_data_temporal_map(data=data_without_label, date_column_name=DATE_COLUMN, kde_resolution=20,
                                              dimensions=DIMENSIONS, period=PERIOD,
                                              dim_reduction=REDUCTION_METHOD, scatter_plot=False, verbose=True)

plot_multivariate_data_temporal_map(data_temporal_map=dtm)

igt_proj_covariate = estimate_igt_projection(dtm,
                                             dimensions=DIMENSIONS,
                                             embedding_type='classicalmds')

plot_IGT_projection(igt_proj_covariate,
                    dimensions=DIMENSIONS,
                    trajectory=True,
                    color_palette=constants.PlotColorPalette.Spectral)

# Concept shift

dtm_concept_shift = estimate_conditional_data_temporal_map(data=dataset_formated, date_column_name=DATE_COLUMN,
                                                           label_column_name=LABEL_NAME, kde_resolution=30,
                                                           dimensions=DIMENSIONS, period=PERIOD,
                                                           dim_reduction=REDUCTION_METHOD, scatter_plot=True,
                                                           verbose=True)

plot_conditional_data_temporal_map(data_temporal_map_dict=dtm_concept_shift)

igt_proj_concept = estimate_igt_projection(dtm_concept_shift,
                                           dimensions=DIMENSIONS,
                                           embedding_type='classicalmds')

plot_IGT_projection(igt_proj_concept,
                    dimensions=DIMENSIONS,
                    trajectory=True,
                    color_palette=constants.PlotColorPalette.Spectral)

# SUPERVISED
metrics = estimate_multibatch_models(data=dataset, inputs_numerical_column_names=NUMERICAL_VARIABLES,
                                     inputs_categorical_column_names=CATEGORICAL_VARIABLES,
                                     output_classification_column_name=LABEL_NAME,
                                     date_column_name=DATE_COLUMN,
                                     period=PERIOD)

plot_multibatch_performance(metrics=metrics, metric_name='F1-SCORE_MACRO')

