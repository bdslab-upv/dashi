import os

import pandas as pd

from dashi.supervised_characterization.estimate_models import estimate_multibatch_models
from dashi.supervised_characterization.plot_performance import plot_multibatch_performance
from dashi.unsupervised_characterization.data_temporal_map.data_temporal_map import (
    estimate_multivariate_data_temporal_map,
    estimate_conditional_data_temporal_map, estimate_univariate_data_temporal_map)
from dashi.unsupervised_characterization.data_temporal_map.data_temporal_map_plotter import \
    plot_univariate_data_temporal_map, \
    plot_multivariate_data_temporal_map, plot_conditional_data_temporal_map
from dashi.unsupervised_characterization.igt.igt_plotting import plot_IGT_projection
from dashi.unsupervised_characterization.igt.igt_projection_estimator import estimate_igt_projection
from dashi.utils import format_data

path = r'C:\Users\David\Desktop\Datasets\COVID_mexico'

data = pd.read_csv(os.path.join(path, 'SAMPLE_FULLCOVIDMEXICO.csv'), low_memory=False)
data = data.drop(columns=['FECHA_ACTUALIZACION', 'ID_REGISTRO', 'FECHA_SINTOMAS', 'FECHA_DEF', 'YEAR'])

# METADATA
DATE_COLUMN = 'FECHA_INGRESO'
CATEGORICAL_VARIABLES = ['ORIGEN', 'SECTOR', 'ENTIDAD_UM', 'SEXO', 'ENTIDAD_NAC', 'ENTIDAD_RES',
                         'MUNICIPIO_RES', 'TIPO_PACIENTE', 'INTUBADO',
                         'NEUMONIA', 'EDAD', 'NACIONALIDAD', 'EMBARAZO', 'HABLA_LENGUA_INDIG',
                         'INDIGENA', 'DIABETES', 'EPOC', 'ASMA', 'INMUSUPR', 'HIPERTENSION',
                         'OTRA_COM', 'CARDIOVASCULAR', 'OBESIDAD', 'RENAL_CRONICA', 'TABAQUISMO',
                         'OTRO_CASO', 'TOMA_MUESTRA_LAB', 'RESULTADO_LAB',
                         'TOMA_MUESTRA_ANTIGENO', 'RESULTADO_ANTIGENO',
                         'CLASIFICACION_FINAL_COVID', 'MIGRANTE', 'PAIS_NACIONALIDAD',
                         'PAIS_ORIGEN', 'UCI', 'RESULTADO_PCR', 'RESULTADO_PCR_COINFECCION',
                         'CLASIFICACION_FINAL_FLU']

LABEL_NAME = 'CLASIFICACION_FINAL_COVID'
DIMENSIONS = 2
PERIOD = 'year'
REDUCTION_METHOD = 'MCA'

dataset_formated = format_data(data, date_column_name=DATE_COLUMN, date_format='%Y-%m-%d',
                               categorical_column_names=CATEGORICAL_VARIABLES)
dataset_without_label = dataset_formated.drop(columns=[LABEL_NAME])

univariate = False
multivariate = False
conditional = True
supervised = True
# UNSUPERVISED
if univariate:
    # univariate
    prob_maps = estimate_univariate_data_temporal_map(dataset_formated, date_column_name=DATE_COLUMN, period=PERIOD,
                                                      numeric_smoothing=True, verbose=True)
    plot_univariate_data_temporal_map(prob_maps[LABEL_NAME])

    univariate_igt_projection = estimate_igt_projection(data_temporal_map=prob_maps[LABEL_NAME],
                                                        dimensions=DIMENSIONS)

    plot_IGT_projection(univariate_igt_projection, trajectory=True, dimensions=DIMENSIONS)

if multivariate:
    # multivariate
    multivariate_dtm = estimate_multivariate_data_temporal_map(dataset_without_label, date_column_name=DATE_COLUMN,
                                                               kde_resolution=30, dimensions=DIMENSIONS, period=PERIOD,
                                                               dim_reduction=REDUCTION_METHOD, scatter_plot=True,
                                                               verbose=True)

    plot_multivariate_data_temporal_map(data_temporal_map=multivariate_dtm)

    multivariate_igt = estimate_igt_projection(data_temporal_map=multivariate_dtm, dimensions=DIMENSIONS)

    plot_IGT_projection(multivariate_igt, dimensions=DIMENSIONS, trajectory=True)

if conditional:
    # conditional multivariate
    conditional_dtm = estimate_conditional_data_temporal_map(data=dataset_formated, date_column_name=DATE_COLUMN,
                                                             label_column_name=LABEL_NAME, kde_resolution=30,
                                                             dimensions=DIMENSIONS, period=PERIOD, scatter_plot=True,
                                                             verbose=True, dim_reduction=REDUCTION_METHOD)

    plot_conditional_data_temporal_map(conditional_dtm)

    conditional_igt_projection = estimate_igt_projection(conditional_dtm, dimensions=DIMENSIONS)

    plot_IGT_projection(conditional_igt_projection, dimensions=DIMENSIONS, trajectory=True)

# SUPERVISED
if supervised:
    data_without_nan = data.dropna(axis=1, how='any')
    INPUT_CATEGORICAL_VARIABLES = ['ORIGEN', 'SECTOR', 'ENTIDAD_UM', 'SEXO', 'ENTIDAD_NAC', 'ENTIDAD_RES',
                                 'MUNICIPIO_RES', 'TIPO_PACIENTE', 'INTUBADO',
                                 'NEUMONIA', 'EDAD', 'NACIONALIDAD', 'EMBARAZO', 'HABLA_LENGUA_INDIG',
                                 'INDIGENA', 'DIABETES', 'EPOC', 'ASMA', 'INMUSUPR', 'HIPERTENSION',
                                 'OTRA_COM', 'CARDIOVASCULAR', 'OBESIDAD', 'RENAL_CRONICA', 'TABAQUISMO',
                                 'OTRO_CASO', 'TOMA_MUESTRA_LAB', 'TOMA_MUESTRA_ANTIGENO',
                                 'RESULTADO_ANTIGENO', 'MIGRANTE',
                                 'PAIS_NACIONALIDAD', 'PAIS_ORIGEN', 'UCI']

    metrics = estimate_multibatch_models(data=data_without_nan,
                                         inputs_categorical_column_names=INPUT_CATEGORICAL_VARIABLES,
                                         output_classification_column_name=LABEL_NAME, date_column_name=DATE_COLUMN,
                                         period=PERIOD, learning_strategy='cumulative')

    plot_multibatch_performance(metrics=metrics, metric_name='RECALL_MACRO')
