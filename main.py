import io

import pandas as pd
import requests

from all_methods import plot_data_temporal_map
from estimate_data_temporal_map import estimate_data_temporal_map
from format_date import format_date

# URL to the dataset
url = 'http://github.com/hms-dbmi/EHRtemporalVariability-DataExamples/raw/master/nhdsSubset.csv'
downloaded_dataset = requests.get(url).content
dataset_string = io.StringIO(downloaded_dataset.decode('utf-8'))

# Read the CSV file
dataframe = pd.read_csv(dataset_string, sep=',', na_values='')

print(dataframe)

#dataframe["age"] = dataframe["age"].astype(float)
dataframe['sex'] = dataframe.sex.astype('category')

print(dataframe['sex'].dtype)

# Display the first few rows of the dataset
pd.set_option('display.max_columns', None)

# Formatted dataset with dates
dataset_formatted = format_date(dataframe, 'date', date_format='%y/%m', verbose=True)

# Testing: counts
#    - string: done (diagcode1)
#    - int: done (age as integer)
#    - categorical:
#    - float: done (age as float) (numericSmoothing = False nice y True da poco distinto por kde), comprobar que se normaliza.

prob_maps = estimate_data_temporal_map(
    data=dataset_formatted,
    date_column_name='date',
    period='month',
    verbose=True,
    numeric_smoothing=False
)


plot_data_temporal_map(
    data_temporal_map=prob_maps['sex'],
    start_value=0,
    end_value=20,
    start_date=min(prob_maps['sex'].dates),
    end_date=max(prob_maps['sex'].dates),
    color_palette='Spectral',
    absolute=True,
    sorting_method='frequency',
    mode='heatmap'
)

