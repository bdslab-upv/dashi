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
       source_column_name='hospital',
       date_format='%Y/%m/%d',
       numerical_column_names=['age', 'weight'],
       categorical_column_names=['gender', 'diagnosis', 'hospital']
   )

2. Unsupervised Temporal Analysis
---------------------------------

Estimate how variable distributions change over time:

.. code-block:: python

   # Univariate analysis
   dtm = ds.estimate_univariate_data_temporal_map(
       data=df,
       date_column_name='date',
       period='month'
   )

   # Plot heatmap
   plot = ds.plot_univariate_data_temporal_map(
       dtm,
       variable_name='weight'
   )
   plot.show()

   # Multivariate analysis with dimensionality reduction
   mv_dtm = ds.estimate_multivariate_data_temporal_map(
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

   plot = ds.plot_multibatch_performance(
        metrics=metrics,
        metric_name='AUC_MACRO'
    )
    plot.show()

To try a simple neural network on the same batches, change ``model_type`` to
``'multilayer_perceptron'`` and optionally configure the sklearn estimator:

.. code-block:: python

   mlp_metrics = ds.estimate_multibatch_models(
       data=df,
       inputs_numerical_column_names=['age', 'weight'],
       inputs_categorical_column_names=['gender'],
       output_classification_column_name='diagnosis',
       date_column_name='date',
       period='month',
       model_type='multilayer_perceptron',
       model_params={'hidden_layer_sizes': (32,), 'max_iter': 300},
   )

Other supported families are ``linear``, ``support_vector_machine``,
``k_nearest_neighbors``, ``extra_trees``, ``random_forest`` and
``histogram_gradient_boosting``. Each training batch fits a fresh sklearn
pipeline; numerical scaling and categorical encoding are learned from that
batch's training rows only, then reused without refitting for test batches.

``model_params`` also configures the existing tree models. For example, use
``model_params={'n_estimators': 200, 'max_depth': 6}`` with
``model_type='random_forest'``, or ``model_params={'max_iter': 150}`` with
``model_type='histogram_gradient_boosting'``. Existing ``n_estimators``,
``max_depth`` and ``max_iter`` arguments still work; conflicting values across
those arguments and ``model_params`` raise an error. Without overrides, the
existing Random Forest (450 trees, depth 9) and HGB (100 iterations) defaults
are unchanged.

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
