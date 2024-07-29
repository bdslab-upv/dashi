import constants
import pandas as pd
from utils import format_date
from sklearn.decomposition import PCA
import numpy as np
from data_temporal_map.data_temporal_map import estimate_multidim_data_temporal_map

dataset = pd.read_csv(r'C:\Users\David\OneDrive - UPV\BDSLab\KINEMAI\text_classifier_python\Emergency_calls\RegresionLogistica\data\99_dim_trimester_mean\embeddings_df.csv')

dataset_formated = format_date(dataset, 'TEMPORAL_TRIMESTER', '%Y-%m')

pca = PCA(n_components=3)
data_pca = pd.DataFrame(pca.fit_transform(dataset_formated.iloc[:, 2:]))
dates = dataset_formated['TEMPORAL_TRIMESTER']

PCA_df = pd.concat([dates, data_pca], axis=1)

estimate_multidim_data_temporal_map(data_pca, dates, binsize=20, period=constants.TEMPORAL_PERIOD_YEAR)
