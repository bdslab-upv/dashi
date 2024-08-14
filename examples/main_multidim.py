import pandas as pd
from sklearn.decomposition import PCA

from datashift import constants
from datashift.data_temporal_map.data_temporal_map import estimate_multidim_data_temporal_map
from datashift.data_temporal_map.data_temporal_map_plotter import plot_data_temporal_map
from datashift.igt.igt_projection_estimator import estimate_igt_projection
from datashift.igt.igt_plotting import plot_IGT_projection


from datashift.utils import format_date

dataset = pd.read_csv(r'C:\Users\David\OneDrive - UPV\BDSLab\KINEMAI\text_classifier_python\Emergency_calls\RegresionLogistica\data\99_dim_trimester_mean\embeddings_df.csv')

dataset_formated = format_date(dataset, 'TEMPORAL_TRIMESTER', '%Y-%m')

pca = PCA(n_components=3)
data_pca = pd.DataFrame(pca.fit_transform(dataset_formated.iloc[:, 2:]))
dates = dataset_formated['TEMPORAL_TRIMESTER']

PCA_df = pd.concat([dates, data_pca], axis=1)

dtm2, dtm3 = estimate_multidim_data_temporal_map(data_pca, dates, binsize=20, period=constants.TEMPORAL_PERIOD_YEAR)

plot_data_temporal_map(dtm2,
                       absolute=False,
                       sorting_method=constants.DataTemporalMapPlotSortingMethod.Frequency)

igt_proj_covariate_2d = estimate_igt_projection(dtm2,
                                                dimensions=2,
                                                embedding_type='classicalmds')

plot_IGT_projection(igt_proj_covariate_2d,
                    dimensions=2,
                    trajectory=True)

igt_proj_covariate_3d = estimate_igt_projection(dtm3,
                                                dimensions=3,
                                                embedding_type='classicalmds')

plot_IGT_projection(igt_proj_covariate_3d,
                    dimensions=3,
                    trajectory=True,
                    color_palette=constants.PlotColorPalette.Viridis)

