import os

import pandas as pd
from dashi.utils import format_data
from dashi.unsupervised_characterization.data_temporal_map.data_temporal_map import (
    estimate_univariate_data_temporal_map, estimate_conditional_data_temporal_map)
from dashi.unsupervised_characterization.data_temporal_map.data_temporal_map_plotter import (
    plot_univariate_data_temporal_map, plot_conditional_data_temporal_map)
from dashi.unsupervised_characterization.igt.igt_projection_estimator import estimate_igt_projection
from dashi.unsupervised_characterization.igt.igt_plotting import plot_IGT_projection
from dashi.supervised_characterization.estimate_models import estimate_multibatch_models
from dashi.supervised_characterization.plot_performance import plot_multibatch_performance

path = r'C:\Users\David\Desktop\Datasets\test_datasets_conceptshifts'

df = pd.read_csv(os.path.join(path, 'generated_dataset.csv'))

DATE_COLUMN = 'date'
NUMERICAL_VARIABLES = ['value']
CATEGORICAL_VARIABLES = ['class']
LABEL_NAME = 'class'
DIMENSIONS = 3
PERIOD = 'month'

univariate = True
conditional = True
supervised = True

df_formated = format_data(df, date_column_name=DATE_COLUMN, date_format='%d/%m/%Y', verbose=True,
                          numerical_column_names=NUMERICAL_VARIABLES, categorical_column_names=CATEGORICAL_VARIABLES)

if univariate:

    prob_maps = estimate_univariate_data_temporal_map(df_formated, date_column_name=DATE_COLUMN, period=PERIOD,
                                                      verbose=True, numeric_smoothing=True)

    plot_univariate_data_temporal_map(prob_maps['value'])

    univariate_igt = estimate_igt_projection(data_temporal_map=prob_maps['value'], dimensions=DIMENSIONS,)

    plot_IGT_projection(univariate_igt, dimensions=DIMENSIONS, trajectory=True)

    # df_1 = df_formated[df_formated['class'] == 'y_1']
    # df_2 = df_formated[df_formated['class'] == 'y_2']
    #
    # prob_maps_1 = estimate_univariate_data_temporal_map(df_1, date_column_name=DATE_COLUMN, period=PERIOD,
    #                                                   verbose=True, numeric_smoothing=True)
    #
    # plot_univariate_data_temporal_map(prob_maps_1['value'])
    #
    # prob_maps_2 = estimate_univariate_data_temporal_map(df_2, date_column_name=DATE_COLUMN, period=PERIOD,
    #                                                   verbose=True, numeric_smoothing=True)
    #
    # plot_univariate_data_temporal_map(prob_maps_2['value'])

if conditional:
    conditional_dtm = estimate_conditional_data_temporal_map(data=df_formated, date_column_name=DATE_COLUMN,
                                                             period=PERIOD, label_column_name=LABEL_NAME, dimensions=1,
                                                             dim_reduction='PCA', verbose=True, kde_resolution=100,
                                                             scatter_plot=True)

    plot_conditional_data_temporal_map(conditional_dtm, absolute=False)

    conditional_igt = estimate_igt_projection(data_temporal_map=conditional_dtm, dimensions=DIMENSIONS)

    plot_IGT_projection(conditional_igt, dimensions=DIMENSIONS, trajectory=True)

if supervised:
    df['class'] = df['class'].astype('category')
    metrics = estimate_multibatch_models(data=df_formated, inputs_numerical_column_names=NUMERICAL_VARIABLES,
                                         output_classification_column_name=LABEL_NAME,
                                         date_column_name=DATE_COLUMN,
                                         period=PERIOD, learning_strategy='cumulative')

    plot_multibatch_performance(metrics=metrics, metric_name='F1-SCORE_MACRO')
