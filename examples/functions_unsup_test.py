"""
DESCRIPTION: main script for testing library functions.
AUTHOR: Pablo Ferri-Borredà
DATE: 18/10/24

"""

# MODULES IMPORT
import plotly.io as pio
import pickle

from dashi.unsupervised_characterization.data_temporal_map.data_temporal_map import estimate_univariate_data_temporal_map, \
    estimate_conditional_data_temporal_map, estimate_multivariate_data_temporal_map
from dashi.unsupervised_characterization.data_temporal_map.data_temporal_map_plotter import plot_univariate_data_temporal_map, \
    plot_conditional_data_temporal_map, plot_multivariate_data_temporal_map
from dashi.unsupervised_characterization.igt.igt_plotting import plot_IGT_projection
from dashi.unsupervised_characterization.igt.igt_projection_estimator import estimate_igt_projection
from dashi.utils import format_data

# SETTINGS
date_column_name = 'DATE'
date_format = '%d/%m/%Y'
period = 'month'
verbose = True
inputs_numerical_column_names = ['PT08.S1(CO)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'RH', 'T', 'C6H6(GT)']
inputs_categorical_column_names = ['AH_LEVEL', 'NO2(GT)_RISK', 'ZONE']
label_column_name = 'NO2(GT)_RISK'
reduction_method = 'FAMD'
pio.renderers.default = 'browser'

# TESTING FLAGS
UNIVARIATE = True
CONDITIONAL = True
MULTIVARIATE = True
IGT = False

# DATA LOADING
with open(r'C:\Users\David\Desktop\Datasets\Datos_Pablo\air_quality_prepared.pkl', 'rb') as f:
    data_prepared = pickle.load(f)

# DATA FORMATTING
dataset_formatted = format_data(
    data_prepared, date_column_name=date_column_name, date_format=date_format, verbose=verbose,
    numerical_column_names=inputs_numerical_column_names,
    categorical_column_names=inputs_categorical_column_names
)

dataset_formatted = dataset_formatted.reset_index(drop=True)

dataset_without_label = dataset_formatted.drop(columns=label_column_name)

if UNIVARIATE:
    # DATA TEMPORAL MAP CALCULATION | UNIVARIATE
    probs_maps_univariate = estimate_univariate_data_temporal_map(
        data=dataset_formatted, date_column_name=date_column_name, period=period, numeric_smoothing=True,
        verbose=verbose
    )

    # DATA TEMPORAL MAP PLOTTING | UNIVARIATE
    plot_univariate_data_temporal_map(probs_maps_univariate['AH_LEVEL'])
    plot_univariate_data_temporal_map(probs_maps_univariate['NO2(GT)_RISK'])
    plot_univariate_data_temporal_map(probs_maps_univariate['T'])
    plot_univariate_data_temporal_map(probs_maps_univariate['C6H6(GT)'])

if CONDITIONAL:
    # DATA TEMPORAL MAP CALCULATION | CONDITIONAL
    probs_maps_conditional = estimate_conditional_data_temporal_map(
        data=dataset_formatted, date_column_name=date_column_name, period=period, label_column_name=label_column_name,
        dim_reduction=reduction_method
    )

    # DATA TEMPORAL MAP PLOTTING | CONDITIONAL
    plot_conditional_data_temporal_map(probs_maps_conditional)

if MULTIVARIATE:
    # DATA TEMPORAL MAP CALCULATION | MULTIVARIATE
    probs_maps_multivariate = estimate_multivariate_data_temporal_map(
        data=dataset_without_label, date_column_name=date_column_name, period=period, dim_reduction=reduction_method,
        verbose=verbose, scatter_plot=True
    )

    # DATA TEMPORAL MAP PLOTTING | MULTIVARIATE
    plot_multivariate_data_temporal_map(probs_maps_multivariate)

if IGT:
    # DATA TEMPORAL MAP CALCULATION | UNIVARIATE
    probs_maps_univariate = estimate_univariate_data_temporal_map(
        data=dataset_formatted, date_column_name=date_column_name, period=period
    )

    # DATA TEMPORAL MAP CALCULATION | MULTIVARIATE
    probs_maps_multivariate = estimate_multivariate_data_temporal_map(
        data=dataset_without_label, date_column_name=date_column_name, period=period, dim_reduction=reduction_method
    )

    probs_maps_conditional = estimate_conditional_data_temporal_map(
        data=dataset_formatted, date_column_name=date_column_name, period=period, label_column_name=label_column_name,
        dim_reduction=reduction_method
    )

    # IGT PROJECTION ESTIMATION
    igt_projection_univ_1 = estimate_igt_projection(
        data_temporal_map=probs_maps_univariate['NO2(GT)_RISK'], dimensions=3, embedding_type='pca'
    )
    igt_projection_univ_2 = estimate_igt_projection(
        data_temporal_map=probs_maps_univariate['T'], dimensions=3, embedding_type='pca'
    )
    igt_projection_multi = estimate_igt_projection(
        data_temporal_map=probs_maps_multivariate, dimensions=3, embedding_type='pca'
    )

    igt_projection_conditional = estimate_igt_projection(
        data_temporal_map=probs_maps_conditional, dimensions=3, embedding_type='pca'
    )

    # IGT PLOTTING
    plot_IGT_projection(
        igt_projection=igt_projection_univ_1, dimensions=3, trajectory=True
    )
    plot_IGT_projection(
        igt_projection=igt_projection_univ_2, dimensions=3, trajectory=True
    )
    plot_IGT_projection(
        igt_projection=igt_projection_multi, dimensions=3, trajectory=True
    )
    plot_IGT_projection(igt_projection=igt_projection_conditional, dimensions=3, trajectory=True)
