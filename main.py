import io

import pandas as pd
import requests

from all_constants import VALID_STRING_TYPE, VALID_DEFAULT_STRING_TYPE
from all_methods import plot_data_temporal_map
from estimate_data_temporal_map import estimate_data_temporal_map
from format_date import format_date

# URL to the dataset
url = 'http://github.com/hms-dbmi/EHRtemporalVariability-DataExamples/raw/master/nhdsSubset.csv'
downloaded_dataset = requests.get(url).content
dataset_string = io.StringIO(downloaded_dataset.decode('utf-8'))

# Read the CSV file
dataframe = pd.read_csv(dataset_string, sep=',', na_values='')

# TODO: pre conversion
# converted_columns = dataframe.loc[:, dataframe.dtypes == VALID_DEFAULT_STRING_TYPE].astype('string')
# dataframe = dataframe.assign(**converted_columns)

# TO TEST
dataframe["age"] = dataframe["age"].astype('float')
dataframe['sex'] = dataframe.sex.astype('category')

# Display the first few rows of the dataset
pd.set_option('display.max_columns', None)

# Formatted dataset with dates
dataset_formatted = format_date(dataframe, 'date', date_format='%y/%m', verbose=True)

# Testing: counts
#    - string: done (diagcode1)
#    - int: done (age as integer)
#    - categorical: (sex as category)
#    - float: done (age as float) (numericSmoothing = False nice y True da poco distinto por kde), comprobar que se normaliza.
# Testing: frequencies
#    - string: done (diagcode1)
#    - int: done  (age as integer)
#    - categorical: done (sex as category)
#    - float: done (age as float) (numericSmoothing = False nice y True da poco distinto por kde), comprobar que se normaliza.


prob_maps = estimate_data_temporal_map(
    data=dataset_formatted,
    date_column_name='date',
    period='month',
    verbose=True,
    numeric_smoothing=False
)

plot_data_temporal_map(
    data_temporal_map=prob_maps['age'],
    start_value=0,
    end_value=20,
    start_date=min(prob_maps['age'].dates),
    end_date=max(prob_maps['age'].dates),
    color_palette='Spectral',
    absolute=False,
    sorting_method='frequency',
    mode='series',
    #plot_title='BULERIA BULERIA, MÁS TE QUIERO CADA DÍA'
)

