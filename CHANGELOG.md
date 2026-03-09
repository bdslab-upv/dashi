# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [Unreleased]
### Planned

- Upcoming feature

## [0.3.0] - 2026-03-09

### Added
- SVD (Singular Value Decomposition) as a new dimensionality reduction method for numerical data.
- Histogram Gradient Boosting as a new model family for both classification and regression tasks in `estimate_multibatch_models`.
- PR-AUC (Precision-Recall Area Under Curve) as a new classification metric, reported per class and as macro average.
- `inplace` parameter in `format_data` function, allowing in-place transformation to save memory on large datasets.
- Support for data type downcasting for memory optimization.

### Changed
- Expanded `plotly` version compatibility from `==5.18.0` to `>=5.18.0,<6.1.0`.
- Expanded `scikit-learn` version compatibility from `==1.5.1` to `>=1.5.1,<2.1.0`.

### Fixed
- Fixed data type recognition when creating supports for variable distribution estimation.
- Fixed bugs in the supervised characterization pipeline that decreased model performance.
- Allowed all datetime units (`datetime64[ns]`, `datetime64[us]`, `datetime64[ms]`, `datetime64[s]`) to be correctly recognized throughout the library.
- Fixed a bug in `estimate_conditional_data_temporal_map` where the `labels_columns` variable was not being filtered and reordered when `start_date` or `end_date` parameters were provided, causing label misalignment with the data after date-based filtering.
- Corrected various warnings that were being incorrectly raised or suppressed.

## [0.1.0] - 2024-12-13
 
First version of the library


  