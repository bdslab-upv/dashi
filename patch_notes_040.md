# dashi v0.4.0 

We are excited to release **dashi v0.4.0**, introducing conditional univariate analysis, JSON serialization and deserialization for dashi maps, and several visualization improvements.

This release extends dashi’s dataset shift characterization capabilities by allowing users to perform conditional univariate analysis for both temporal and source-based maps. It also improves the portability, storage, and reuse of dashi analysis results through JSON serialization and object reconstruction utilities.

## What's New

### Conditional Univariate Analysis

dashi now supports **conditional univariate analysis** for both temporal and source-based data characterization.

This new functionality allows users to analyze the behavior of individual variables conditioned on class or group information, making it easier to inspect how distributions evolve across time or data sources for each class independently.

The following estimation functions have been added:

- `estimate_conditional_univariate_data_temporal_map`
- `estimate_conditional_univariate_data_source_map`

The corresponding plotting functions have also been implemented:

- `plot_conditional_univariate_data_temporal_map`
- `plot_conditional_univariate_data_source_map`

These additions expand dashi’s conditional analysis capabilities and provide a more detailed view of univariate dataset shift patterns across temporal and source dimensions.

### JSON Serialization and Deserialization

A new serialization and deserialization functionality has been implemented for dashi temporal and source maps.

This allows users to export dashi objects into JSON files and later reconstruct them back into dashi objects, enabling easier storage, sharing, transportability, and reproducibility of analysis results.

The main functions are:

- `to_json`
- `from_json`

In addition, this release includes utilities to transform dashi map objects into dictionaries and reconstruct them back from dictionaries.

#### Source map utilities

- `data_source_map_to_dict`
- `dict_to_data_source_map`
- `multivariate_data_source_map_to_dict`
- `dict_to_multivariate_data_source_map`
- `conditional_source_map_to_dict`
- `dict_to_conditional_source_map`
- `conditional_univariate_source_map_to_dict`
- `dict_to_conditional_univariate_source_map`

#### Temporal map utilities

- `data_temporal_map_to_dict`
- `dict_to_data_temporal_map`
- `multivariate_data_temporal_map_to_dict`
- `dict_to_multivariate_data_temporal_map`
- `conditional_temporal_map_to_dict`
- `dict_to_conditional_temporal_map`
- `conditional_univariate_temporal_map_to_dict`
- `dict_to_conditional_univariate_temporal_map`

These utilities make it easier to integrate dashi outputs into external workflows, store intermediate results, share analyses between environments, and reload previous characterization results without recomputing the full analysis. The documentation's quickstart section shows an example of how to use this functionality. 

### Shared Y-Axis Control in Conditional Source Maps

The following plotting functions now include a new `shared_yaxis` argument:

- `plot_conditional_univariate_data_source_map`
- `plot_conditional_data_source_map`

This argument allows users to control whether all class-specific maps share the same y-axis or use independent y-axis scales.

This provides more flexibility when visualizing conditional source maps, especially when class distributions have different value ranges.

## Modifications and Improvements

### Variable Selection in Univariate Map Plotting

The following plotting functions now include a new `variable_name` argument:

- `plot_univariate_data_temporal_map`
- `plot_univariate_data_source_map`

When a dictionary containing maps for multiple variables is passed to these functions, the `variable_name` argument allows users to select the specific variable to plot.

This improves usability when working with multiple univariate maps and makes the plotting workflow more explicit and flexible.

### Improved Representation of Categorical Variables in Source Maps

The representation of categorical variables in data source maps has been improved.

Previously, categorical variables were displayed using continuous lines, which was not appropriate for discrete categories. These cases are now represented using bar plots, providing a clearer and more meaningful visualization for categorical data.

### Visualization Fixes

Several bugs related to subplot spacing and overlapping have been fixed.

These improvements provide cleaner figures and better readability, particularly when generating complex maps with multiple subplots or class-specific visualizations.

## Bug Fixes and Stability Improvements

This release incorporates critical robustness improvements and bug fixes across supervised and unsupervised modules:

### Supervised Characterization
- **Small Batch Stability**: Addressed an issue where `HistGradientBoosting` models crashed during internal validation splits on small batches by dynamically disabling early stopping when necessary.
- **Missing Class Probabilities**: Fixed crashes in classification metrics when some classes were entirely absent in a batch by dynamically preserving the prediction probability shape.
- **Robust Metric Calculations**: Fixed `log_loss` evaluations that failed on missing labels by explicitly anchoring to the global class list.
- **Performance Heatmap Colors**: Resolved a rendering bug in error metric heatmaps (e.g., Mean Squared Error) by leveraging Plotly's native `reversescale`, properly mapping lowest errors to green and highest to red.

### Unsupervised Characterization & Data Maps
- **Data Map Count Alignments**: Fixed statistical distribution skewing in both Temporal and Source Maps by strictly enforcing chronological and source-appearance order alignments between batch counts and Kernel Density Estimates.
- **Constant Variables and Dimensionality Checks**: Resolved `div-by-zero` crashes when processing constant variables. Additionally, fixed index out of bounds exceptions and strengthened dimensionality checks across Multivariate Maps to prevent silent failures.
- **IGT Trajectory Stability**: Prevented `UnivariateSpline` evaluation crashes when tracking only 2 or 3 temporal batches by dynamically lowering the spline degree.
- **IGT Chronological Colors**: Fixed a visual bug in trajectory plotting where data point colors were previously mapped to day-of-week rather than the correct yearly week index.

### Data Formatting & Core Utils
- **Missing Temporal Data Handling**: Implemented explicit pruning of `NaT` (Not a Time) rows to prevent the generation of invalid temporal batches.
- **Formatting Match Optimizations**: Refactored date format string matching to use more efficient and readable expressions.

## Installation

```bash
pip install --upgrade dashi