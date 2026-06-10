Quick Start Guide
=================

This guide walks you through the main features of dashi.

1. Data Formatting
------------------

Before any analysis, format your DataFrame so that dates and types are correct:

.. code-block:: python

   import pandas as pd
   import dashi as ds

   df = pd.read_csv('my_data.csv')

   df = ds.format_data(
       df,
       date_column_name='date',
       date_format='%Y/%m/%d',
       numerical_column_names=['age', 'weight'],
       categorical_column_names=['gender', 'diagnosis']
   )

2. Unsupervised Temporal Analysis
---------------------------------

Estimate how variable distributions change over time:

.. code-block:: python

   # Univariate analysis
   dtm = ds.estimate_univariate_data_temporal_map(
       data=df,
       date_column='date',
       period='month'
   )

   # Plot heatmap
   plot = ds.plot_univariate_data_temporal_map(
       dtm,
       variable_name='weight'
   )
   plot.show()

   # Multivariate analysis with dimensionality reduction
   mv_dtm = dashi.estimate_multivariate_data_temporal_map(
       data=df,
       date_column_name='date',
       period='month',
       dim_reduction='FAMD',
       dimensions=2
   )

   # Plot heatmap
   plot = ds.plot_multivariate_data_temporal_map(mv_dtm)
   plot.show()


3. Unsupervised Multi-Source Analysis
-------------------------------------

Compare distributions across different data sources:

.. code-block:: python

   dsm = ds.estimate_univariate_data_source_map(
       data=df,
       source_column='hospital'
   )

   plot = ds.plot_univariate_data_source_map(
       dsm,
       variable_name='weight'
   )
   plot.show()

4. Variability Metrics (IGT & MSV)
-----------------------------------

Quantify temporal or source variability:

.. code-block:: python

   # Information Geometric Temporal (IGT) projection
   igt = ds.estimate_igt_projection(dtm, embedding_type='classicalmds')
   plot = ds.plot_IGT_projection(igt)
   plot.show()

   # Multi-Source Variability (MSV) metrics
   msv = ds.estimate_MSV_metrics(dsm)
   plot = ds.plot_MSV(msv)
   plot.show()

5. Supervised Characterization
------------------------------

Evaluate model performance across temporal or source batches:

.. code-block:: python

   metrics = ds.estimate_multibatch_models(
       data=df,
       inputs_numerical_column_names=['age', 'weight'],
       inputs_categorical_column_names=['gender'],
       output_classification_column_name='diagnosis',
       date_column_name='date',
       period='month',
       learning_strategy='from_scratch',
       model_type='histogram_gradient_boosting'
   )

   plot = ds.plot_performance(
       metrics,
       metric_name='ROC-AUC_MACRO
   )
   plot.show()

6. Exporting Temporal Maps to JSON
-----------------------------------

Temporal map objects can be converted to JSON-compatible dictionaries, exported as JSON strings, and reconstructed later:

.. code-block:: python

   from dashi.serialization import to_json, from_json
   from dashi.serialization.temporal import (
       data_temporal_map_to_dict,
       dict_to_data_temporal_map,
   )

   dtm = ds.estimate_univariate_data_temporal_map(
       data=df,
       date_column_name='date',
       period='month'
   )

   json_payload = to_json(data_temporal_map_to_dict(dtm))
   reconstructed_dtm = dict_to_data_temporal_map(from_json(json_payload))

For conditional univariate temporal maps, use the matching conditional helpers:

.. code-block:: python

   from dashi.serialization.temporal import (
       conditional_univariate_temporal_map_to_dict,
       dict_to_conditional_univariate_temporal_map,
   )

   conditional_dtm = ds.estimate_conditional_univariate_data_temporal_map(
       data=df,
       date_column_name='date',
       label_column_name='diagnosis',
       period='month'
   )

   json_payload = to_json(conditional_univariate_temporal_map_to_dict(conditional_dtm))
   reconstructed_conditional_dtm = dict_to_conditional_univariate_temporal_map(from_json(json_payload))
