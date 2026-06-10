dashi documentation
===================

**dashi** is a flexible and powerful Python toolkit for dataset shift analysis and characterization,
providing supervised and unsupervised evaluation of temporal and multi-source data shifts,
visualization tools, and statistical insights for data integrity and model performance monitoring.

.. note::
   Install dashi with: ``pip install dashi``

Getting Started
---------------

.. code-block:: python

   import dashi as ds

   # Format your data
   df = ds.format_data(
       df,
       date_column_name='date',
       date_format='%Y/%m/%d',
       numerical_column_names=['age', 'weight'],
       categorical_column_names=['gender', 'diagnosis']
   )

   # Estimate a univariate Data Temporal Map
   dtm = ds.estimate_univariate_data_temporal_map(
       data=df,
       date_column='date',
       period='month'
   )

   # Plot the results
   plot = ds.plot_data_temporal_map(dtm['weight'])


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/installation
   guide/quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/data_formatting
   api/unsupervised_temporal
   api/unsupervised_source
   api/unsupervised_variability
   api/supervised
   api/serialization

.. toctree::
   :maxdepth: 1
   :caption: About

   changelog
