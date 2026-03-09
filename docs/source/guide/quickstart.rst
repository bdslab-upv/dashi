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
   plot = ds.plot_univariate_data_temporal_map(dtm['weight'])

   # Multivariate analysis with dimensionality reduction
   mv_dtm = dashi.estimate_multivariate_data_temporal_map(
       data=df,
       date_column_name='date',
       period='month',
       dim_reduction='PCA',
       dimensions=2
   )

   # Plot heatmap
   plot = ds.plot_multivariate_data_temporal_map(mv_dtm)


3. Unsupervised Multi-Source Analysis
-------------------------------------

Compare distributions across different data sources:

.. code-block:: python

   dsm = ds.estimate_univariate_data_source_map(
       data=df,
       source_column='hospital'
   )

   plot = ds.plot_univariate_data_source_map(dsm['weight'])

4. Variability Metrics (IGT & MSV)
-----------------------------------

Quantify temporal or source variability:

.. code-block:: python

   # Information Geometric Temporal (IGT) projection
   igt = ds.estimate_igt_projection(dtm, embedding_type='classicalmds')
   plot = ds.plot_IGT_projection(igt)

   # Multi-Source Variability (MSV) metrics
   msv = ds.estimate_MSV_metrics(dsm)
   plot = ds.plot_MSV(msv)

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

   performance_df = ds.arrange_performance_metrics(
       metrics=metrics,
       metric_name='AUC_MACRO'
   )

   plot = ds.plot_performance(
   performance_df,
   metric_name='ROC-AUC_MACRO
   )