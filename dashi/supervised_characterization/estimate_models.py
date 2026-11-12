# Copyright 2024 Biomedical Data Science Lab, Universitat Politècnica de València (Spain)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Main function for estimating models over multiple temporal or multi-source batches.
"""

import gc
import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn.metrics as skmet
from numpy import ndarray, sqrt
from pandas import DataFrame
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.svm import SVC, SVR
from tqdm.auto import tqdm

__all__ = ['estimate_multibatch_models']

_NEW_MODEL_TYPES = (
    'linear', 'support_vector_machine', 'k_nearest_neighbors',
    'extra_trees', 'multilayer_perceptron',
)
_MODEL_TYPES = ('random_forest', 'histogram_gradient_boosting') + _NEW_MODEL_TYPES


def _resolve_model_params(model_type: str, model_params: Optional[dict], *,
                          max_iter: Optional[int], n_estimators: Optional[int],
                          max_depth: Optional[int]) -> dict:
    """Merge legacy arguments with estimator parameters without changing defaults."""
    if model_params is not None:
        if type(model_params) is not dict or any(type(key) is not str for key in model_params):
            raise TypeError('model_params must be a dictionary with string keys.')
    params = dict(model_params or {})

    defaults = {'max_iter': 100, 'n_estimators': 450, 'max_depth': 9}
    owners = {'max_iter': 'histogram_gradient_boosting',
              'n_estimators': 'random_forest', 'max_depth': 'random_forest'}
    for name, value in (('max_iter', max_iter), ('n_estimators', n_estimators), ('max_depth', max_depth)):
        if value is not None:
            if type(value) is not int:
                raise TypeError(f'{name} must be an integer.')
            if value <= 0:
                raise ValueError(f'{name} must be greater than 0.')
        if model_type == owners[name]:
            if value is not None and name in params and params[name] != value:
                raise ValueError(f'Conflicting values for {name}: legacy argument and model_params.')
            params.setdefault(name, defaults[name] if value is None else value)
        elif value is not None and value != defaults[name]:
            raise ValueError(f'{name} is only valid when model_type is {owners[name]}; '
                             'use model_params for other model families.')

    # These parameters are calculated separately for each training batch.
    if model_type == 'histogram_gradient_boosting':
        protected = {'categorical_features', 'early_stopping', 'scoring'} & params.keys()
        if protected:
            raise ValueError(f'HGB batch settings cannot be supplied in model_params: {sorted(protected)}.')
    return params


def _build_preprocessor(model_type: str, num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Create an unfitted, training-only feature transformer for a model family."""
    transformers = []
    # Keep categorical positions first for HGB's native categorical splits.
    if cat_cols:
        if model_type == 'histogram_gradient_boosting':
            encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value', unknown_value=np.nan,
                encoded_missing_value=np.nan, dtype=np.float32,
            )
        else:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformers.append(('categorical', encoder, cat_cols))
    if num_cols:
        transformers.append(('numerical', RobustScaler(), num_cols))
    return ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)


def _build_model(model_type: str, *, is_classification: bool, y_train: ndarray,
                 batch_id: str, n_categorical: int, model_params: dict):
    """Create a fresh estimator; HGB's internal validation uses training rows only."""
    random_seed = 42
    if model_type == 'histogram_gradient_boosting':
        early_stopping = _hgb_early_stopping_is_safe(y_train, is_classification=is_classification)
        if not early_stopping:
            split_type = 'stratified validation' if is_classification else 'internal validation'
            warnings.warn(f"Disabling early stopping for batch '{batch_id}' because it is too "
                          f'small to form a valid {split_type} split.')
        common = dict(max_iter=100, random_state=random_seed, early_stopping=early_stopping,
                      scoring='loss', categorical_features=list(range(n_categorical)) if n_categorical else None)
        if is_classification:
            estimator = HistGradientBoostingClassifier(class_weight='balanced', **common)
        else:
            estimator = HistGradientBoostingRegressor(**common)
    elif model_type == 'random_forest':
        common = dict(n_estimators=450, max_depth=9,
                      random_state=random_seed, n_jobs=-1)
        if is_classification:
            estimator = RandomForestClassifier(class_weight='balanced', **common)
        else:
            estimator = RandomForestRegressor(**common)
    else:
        constructors = {
            'linear': (LogisticRegression, Ridge),
            'support_vector_machine': (SVC, SVR),
            'k_nearest_neighbors': (KNeighborsClassifier, KNeighborsRegressor),
            'extra_trees': (ExtraTreesClassifier, ExtraTreesRegressor),
            'multilayer_perceptron': (MLPClassifier, MLPRegressor),
        }
        classifier, regressor = constructors[model_type]
        defaults = {
            'linear': (dict(max_iter=1000, random_state=random_seed), {}),
            'support_vector_machine': (dict(random_state=random_seed), {}),
            'k_nearest_neighbors': (dict(n_neighbors=min(5, len(y_train))),
                                    dict(n_neighbors=min(5, len(y_train)))),
            'extra_trees': (dict(random_state=random_seed, n_jobs=-1),
                            dict(random_state=random_seed, n_jobs=-1)),
            'multilayer_perceptron': (dict(random_state=random_seed), dict(random_state=random_seed)),
        }
        estimator = (classifier if is_classification else regressor)(
            **defaults[model_type][0 if is_classification else 1])
    estimator.set_params(**model_params)
    if model_type == 'support_vector_machine' and is_classification:
        # Calibration folds are drawn exclusively from the current training batch.
        return CalibratedClassifierCV(estimator, cv=2, ensemble=False)
    return estimator


def _hgb_early_stopping_is_safe(y_train: ndarray, *, is_classification: bool,
                                validation_fraction: float = 0.1) -> bool:
    """
    Decide whether HistGradientBoosting early stopping can be used on a batch.

    Early stopping performs an internal validation split (stratified for
    classification). On very small batches this split is invalid and sklearn
    raises at fit time, so early stopping is disabled in those cases.
    """
    n_samples = len(y_train)
    if n_samples < 2:
        return False

    # sklearn draws a validation set of ceil(validation_fraction * n_samples)
    n_validation = int(np.ceil(validation_fraction * n_samples))
    if n_validation < 1 or n_samples - n_validation < 1:
        return False

    if is_classification:
        _, counts = np.unique(y_train, return_counts=True)
        # The validation set must be able to hold every class, and the stratified
        # split needs at least two samples per class to populate train and validation.
        if n_validation < len(counts) or counts.min() < 2:
            return False

    return True


# FUNCTION DEFINITION
def estimate_multibatch_models(
    *,
    data: pd.DataFrame,
    inputs_numerical_column_names: Optional[List[str]] = None,
    inputs_categorical_column_names: Optional[List[str]] = None,
    output_regression_column_name: Optional[str] = None,
    output_classification_column_name: Optional[str] = None,
    date_column_name: Optional[str] = None,
    period: Optional[str] = None,
    source_column_name: Optional[str] = None,
    learning_strategy: Optional[str] = "from_scratch",
    model_type: Optional[str] = 'histogram_gradient_boosting',
    max_iter: Optional[int] = None,
    n_estimators: Optional[int] = None,
    max_depth: Optional[int] = None,
    model_params: Optional[dict] = None,
    shuffle: bool = False,
) -> Dict[Tuple, Dict[str, float]]:
    """
    Estimates models across multiple batches, based on either time (temporal) or source.
    Requires specifying one target variable (regression or classification) and at least one
    numerical or categorical input feature within the input DataFrame.
    At the same time, it is necessary to provide either a date variable (indicating the period with the corresponding
    argument) or a source variable. The date variable must be a valid date, and the source variable categories need to
    be specified as strings.
    Additionally, it is recommended that the amount of data in each batching group be sufficient for statistical
    representativeness.

    Parameters
    ----------
    data : DataFrame
        The input data containing numerical and/or categorical features, as well as the target variable
        (either a classification or regression target).

    inputs_numerical_column_names : Optional[List[str]], default=None
        List of column names representing numerical input features. If there are no numerical input features,
        set this to None.

    inputs_categorical_column_names : Optional[List[str]], default=None
        List of column names representing categorical input features. If there are no categorical input features,
        set this to None.

    output_regression_column_name : Optional[str], default=None
        Column name for the regression target variable. If there is no regression target, set this to None.

    output_classification_column_name : Optional[str], default=None
        Column name for the classification target variable. If there is no classification target, set this to None.

    date_column_name : Optional[str], default=None
        Column name containing date or time information for temporal batching. If performing source-based
        analysis instead of temporal batching, set this to None.

    period : Optional[str], default=None
        Period for batching the data ('month' or 'year') when using temporal batching. If not using temporal
        batching, set this to None.

    source_column_name : Optional[str], default=None
        Column name representing the source of the data (for multi-source batching). If performing temporal
        batching, set this to None.

    learning_strategy : Optional[str], default='from_scratch'
        Defines the learning strategy: either 'from_scratch' or 'cumulative'. Note that the 'cumulative' strategy
        can only be applied to temporal analyses, not multi-source analyses.

    model_type : Optional[str], default='histogram_gradient_boosting'
        Model family, selecting the corresponding scikit-learn estimator for the
        requested task:

        - ``'random_forest'``: ``RandomForestClassifier`` or ``RandomForestRegressor``
        - ``'histogram_gradient_boosting'``: ``HistGradientBoostingClassifier`` or
          ``HistGradientBoostingRegressor``
        - ``'linear'``: ``LogisticRegression`` or ``Ridge``
        - ``'support_vector_machine'``: calibrated ``SVC`` or ``SVR``
        - ``'k_nearest_neighbors'``: ``KNeighborsClassifier`` or
          ``KNeighborsRegressor``
        - ``'extra_trees'``: ``ExtraTreesClassifier`` or ``ExtraTreesRegressor``
        - ``'multilayer_perceptron'``: ``MLPClassifier`` or ``MLPRegressor``

        Each training batch fits its own preprocessing and model. HGB uses
        ordinal-encoded native categorical features; other families use one-hot
        encoding with unknown test categories ignored.

    max_iter : int, optional
        Legacy HGB boosting-iteration setting (effective default 100). Use
        ``model_params={'max_iter': ...}`` for new calls.

    n_estimators : int, optional
        Legacy Random Forest tree count (effective default 450). Use
        ``model_params={'n_estimators': ...}`` for new calls.

    max_depth : int, optional
        Legacy Random Forest tree depth (effective default 9). Use
        ``model_params={'max_depth': ...}`` for new calls.

    model_params : dict, optional
        sklearn estimator parameters for any model family. Keys must be valid
        parameters of the selected estimator. Different values for the same
        setting in a legacy argument and this dictionary raise ``ValueError``.
        HGB's categorical features, early stopping and scoring are managed per
        batch. SVC probabilities are calibrated
        with two training-only folds; its ``probability`` parameter is managed
        internally and cannot be passed here.
        For the multilayer perceptron, sklearn's default architecture and
        ``max_iter=200`` apply unless overridden; convergence warnings may occur.

    shuffle : bool, default=False
        How the held-out test set of each batch (20% of its rows) is chosen.

        - ``False`` (default): the last 20% of the batch rows, after sorting by date for temporal
          batching or in the input order for source batching. With dates finer than the batching
          period, the test set therefore holds only the latest dates of each batch (for example,
          October to December with yearly batches of monthly data).
        - ``True``: a random 20% of the batch rows (fixed seed 42). For classification, the split
          is stratified by label when every class of the batch has at least two rows.

    Returns
    -------
    Dict[Tuple, Dict[str, float]]
        A dictionary where each key is a tuple `(train_batch_ids, test_batch_id, 'test')`
        representing the training/testing combination. Each corresponding value is another
        dictionary containing the calculated performance metrics for that specific test.

        The inner dictionary contains metric names (str) and their values (float):
        Regression metrics, if applicable:
            - 'MEAN_ABSOLUTE_ERROR'
            - 'MEAN_SQUARED_ERROR'
            - 'ROOT_MEAN_SQUARED_ERROR'
            - 'R_SQUARED'
        Classification metrics, if applicable:
            - 'AUC_{class_identifier}'
            - 'PR-AUC_{class_identifier}'
            - 'AUC_MACRO'
            - 'PR-AUC_MACRO'
            - 'LOGLOSS'
            - 'RECALL_{class_identifier}'
            - 'PRECISION_{class_identifier}'
            - 'F1-SCORE_{class_identifier}'
            - 'ACCURACY'
            - 'RECALL_MACRO'
            - 'RECALL_MICRO'
            - 'RECALL_WEIGHTED'
            - 'PRECISION_MACRO'
            - 'PRECISION_MICRO'
            - 'PRECISION_WEIGHTED'
            - 'F1-SCORE_MACRO'
            - 'F1-SCORE_MICRO'
            - 'F1-SCORE_WEIGHTED'
    """

    if not isinstance(shuffle, bool):
        raise TypeError('shuffle must be a boolean.')

    resolved_model_params = _check_inputs(
        data=data,
        inputs_numerical_column_names=inputs_numerical_column_names,
        inputs_categorical_column_names=inputs_categorical_column_names,
        output_regression_column_name=output_regression_column_name,
        output_classification_column_name=output_classification_column_name,
        date_column_name=date_column_name,
        period=period,
        source_column_name=source_column_name,
        learning_strategy=learning_strategy,
        model_type=model_type,
        max_iter=max_iter,
        n_estimators=n_estimators,
        max_depth=max_depth,
        model_params=model_params,
    )

    # Shallow copy to avoid modifying the original DataFrame
    df_work = data.copy(deep=False)

    # Label encoding for y (classification)
    index2class_map = None
    if output_classification_column_name is not None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(df_work[output_classification_column_name])
        df_work[output_classification_column_name] = y_encoded.astype("int64")
        index2class_map = dict(enumerate(le.classes_))

    # Batching column
    if date_column_name is None and source_column_name is not None:
        batching_column_name = source_column_name
    elif date_column_name is not None and source_column_name is None:
        df_work[date_column_name] = pd.to_datetime(df_work[date_column_name], errors="coerce")
        df_work = df_work.dropna(subset=[date_column_name])
        df_work = df_work.sort_values(by=date_column_name)

        if period == "month":
            df_work[date_column_name] = df_work[date_column_name].dt.strftime("%B %Y")
        elif period == "year":
            df_work[date_column_name] = df_work[date_column_name].dt.strftime("%Y")
        else:
            raise ValueError("Period must be 'month' or 'year'.")
        batching_column_name = date_column_name
    else:
        raise ValueError(
            "Invalid configuration: Please provide exactly one of 'date_column_name' for "
            "temporal batching or 'source_column_name' for source-based batching. "
            "You have either provided both or neither."
        )

    # Generate split indexes based on batching
    split_indexes = _generate_split_indexes(
        data=df_work,
        batching_column_name=batching_column_name,
        shuffle=shuffle,
        stratify_column_name=output_classification_column_name,
    )
    # Extract batch identifiers
    batch_identifiers = list(split_indexes.keys())

    cat_cols = inputs_categorical_column_names or []
    num_cols = inputs_numerical_column_names or []

    if len(cat_cols) + len(num_cols) == 0:
        raise ValueError("No input features provided.")

    # ---------- Main loop ----------

    metrics: Dict[Tuple, Dict[str, float]] = {}

    train_loop = tqdm(
        enumerate(batch_identifiers),
        total=len(batch_identifiers),
        colour="#32CD32",
        desc="Learning and testing over experiences",
    )

    for idx, batch_idf_train in train_loop:

        # --- Resolve training row positions ---
        if learning_strategy == "from_scratch":
            current_train_batch_ids = (batch_idf_train,)
            train_positions = split_indexes[batch_idf_train]["train_test"]["train_indexes"]
        elif learning_strategy == "cumulative":
            current_train_batch_ids = tuple(batch_identifiers[i] for i in range(0, idx + 1))
            train_positions = np.concatenate(
                [split_indexes[bid]["train_test"]["train_indexes"] for bid in current_train_batch_ids]
            )
        else:
            raise ValueError("Unrecognized learning strategy.")

        train_sub = df_work.iloc[train_positions]
        is_classification = output_classification_column_name is not None
        target = output_classification_column_name if is_classification else output_regression_column_name
        y_train = train_sub[target].to_numpy(copy=False)
        if (is_classification and model_type in ('linear', 'support_vector_machine')
                and len(np.unique(y_train)) < 2):
            raise ValueError(f"Training batch '{batch_idf_train}' must contain at least two classes.")
        if is_classification and model_type == 'support_vector_machine':
            _, class_counts = np.unique(y_train, return_counts=True)
            if class_counts.min() < 2:
                raise ValueError(f"Training batch '{batch_idf_train}' needs at least two samples "
                                 'per class for SVC probability calibration.')

        model = _build_model(
            model_type, is_classification=is_classification, y_train=y_train,
            batch_id=batch_idf_train, n_categorical=len(cat_cols),
            model_params=resolved_model_params,
        )
        pipeline = Pipeline([
            ('preprocess', _build_preprocessor(model_type, num_cols, cat_cols)),
            ('float32', FunctionTransformer(np.asarray, kw_args={'dtype': np.float32})),
            ('model', model),
        ])
        pipeline.fit(train_sub, y_train)

        # Free training buffers
        del train_sub, y_train
        gc.collect()

        # --- Testing loop ---
        for batch_idf_test in batch_identifiers:
            test_key = (current_train_batch_ids, batch_idf_test, "test")

            test_positions = split_indexes[batch_idf_test]["train_test"]["test_indexes"]
            test_sub = df_work.iloc[test_positions]

            if output_regression_column_name is not None:
                y_test = test_sub[output_regression_column_name].to_numpy(copy=False)
                y_pred = pipeline.predict(test_sub)
                metrics[test_key] = _get_regression_metrics(y_true=y_test, y_pred=y_pred)
            else:
                y_test = test_sub[output_classification_column_name].to_numpy(copy=False)
                probs_hat = pipeline.predict_proba(test_sub)
                labels_hat = pipeline.predict(test_sub)

                # Pad probabilities for missing classes
                probs_hat_full = np.zeros((len(y_test), len(index2class_map)), dtype=np.float32)
                for col_idx, cls_val in enumerate(pipeline.classes_):
                    probs_hat_full[:, cls_val] = probs_hat[:, col_idx]

                metrics_pre = _get_presaturation_classification_metrics(
                    label_true=y_test,
                    label_scores=probs_hat_full,
                    index2class_map=index2class_map,
                )
                metrics_post = _get_postsaturation_classification_metrics(
                    label_true=y_test,
                    label_predicted=labels_hat,
                    index2class_map=index2class_map,
                )
                metrics[test_key] = {**metrics_pre, **metrics_post}

            del test_sub
            gc.collect()

        # Free model + preprocessors
        del pipeline, model
        gc.collect()

    return metrics

# INPUTS CHECKING
def _check_inputs(
        *,
        data: DataFrame,
        inputs_numerical_column_names: Optional[List[str]] = None,
        inputs_categorical_column_names: Optional[List[str]] = None,
        output_regression_column_name: Optional[str] = None,
        output_classification_column_name: Optional[str] = None,
        date_column_name: Optional[str] = None,
        period: Optional[str] = None,
        source_column_name: Optional[str] = None,
        learning_strategy: Optional[str] = 'from_scratch',
        model_type: Optional[str],
        max_iter: Optional[int] = None,
        n_estimators: Optional[int] = None,
        max_depth: Optional[int] = None,
        model_params: Optional[dict] = None,
) -> dict:
    """
    Validate the inputs provided for model estimation.

    Parameters
    ----------
    data : DataFrame
        The input data containing features and target variables.

    inputs_numerical_column_names : Optional[List[str]], default=None
        List of column names representing numerical input features, if applicable.

    inputs_categorical_column_names : Optional[List[str]], default=None
        List of column names representing categorical input features, if applicable.

    output_regression_column_name : Optional[str], default=None
        Column name for the regression target variable, if applicable.

    output_classification_column_name : Optional[str], default=None
        Column name for the classification target variable, if applicable.

    date_column_name : Optional[str], default=None
        Column name containing date or time information for temporal batching, if applicable.

    period : Optional[str], default=None
        Period for batching the data ('month' or 'year') when using temporal batching.

    source_column_name : Optional[str], default=None
        Column name representing the source of the data (for multi-source batching).

    learning_strategy : Optional[str], default='from_scratch'
        Defines the learning strategy: 'from_scratch' or 'cumulative'.

    model_type : Optional[str], default='histogram_gradient_boosting'
        Model family selected for batched training and evaluation.

    max_iter : int, optional
        Legacy HGB boosting-iteration setting (effective default 100).

    n_estimators : int, optional
        Legacy Random Forest tree count (effective default 450).

    max_depth : int, optional
        Legacy Random Forest tree depth (effective default 9).

    model_params : dict, optional
        Estimator-specific parameters for all model families.

    Returns
    -------
    dict
        Resolved estimator parameters, including the existing RF/HGB defaults.

    Raises
    ------
    TypeError
        If any input parameters have an incorrect type.
    ValueError
        If any input parameters are invalid or inconsistent.
    """

    # Data
    if type(data) is not DataFrame:
        raise TypeError('Data must be encapsulated into a Data frame object.')
    else:
        if data.isnull().values.any():
            raise ValueError('Missing data is present in your data frame object. '
                             'Please, process them before calling this function.')

    # Date column
    if date_column_name is not None:
        if type(date_column_name) is not str:
            raise TypeError('Date column must be specified as a string.')
        if date_column_name not in data.columns:
            raise ValueError('Date column not found in the current data frame.')
        # batching period
        if period is None:
            raise ValueError("A batching period needs to be specified: either 'month' or 'year'.")
        else:
            if period not in ('month', 'year'):
                raise ValueError("Current supported batching periods are 'month' and 'year'.")

    # Source column
    if source_column_name is not None:
        if type(source_column_name) is not str:
            raise TypeError('Source column must be specified as a string.')
        if source_column_name not in data.columns:
            raise ValueError('Source column not found in the current data frame.')

    # Date and source column
    if date_column_name is None and source_column_name is None:
        raise ValueError('Either the date column or the source column needs to the provided.')
    if date_column_name is not None and source_column_name is not None:
        raise ValueError('Just one batching column can be considered (date or source but not both simultaneously).')

    # Inputs numerical columns names
    if inputs_numerical_column_names is not None:
        if type(inputs_numerical_column_names) is not list:
            raise TypeError('Numerical inputs columns need to be encapsulated in a list.')
        else:
            if len(inputs_numerical_column_names) == 0:
                raise ValueError('Numerical inputs column names list is void.')
            else:
                for inp_num_col in inputs_numerical_column_names:
                    if type(inp_num_col) is not str:
                        raise TypeError('Numerical input column must be specified as a string.')
                    else:
                        if inp_num_col not in data.columns:
                            raise ValueError('Numerical input column not found in the current data frame.')

    # Inputs categorical columns names
    if inputs_categorical_column_names is not None:
        if type(inputs_categorical_column_names) is not list:
            raise TypeError('Categorical inputs columns need to be encapsulated in a list.')
        else:
            if len(inputs_categorical_column_names) == 0:
                raise ValueError('Categorical inputs column names list is void.')
            else:
                for inp_cat_col in inputs_categorical_column_names:
                    if type(inp_cat_col) is not str:
                        raise TypeError('Categorical input column must be specified as a string.')
                    else:
                        if inp_cat_col not in data.columns:
                            raise ValueError('Categorical input column not found in the current data frame.')

    # Inputs numerical columns names and inputs categorical columns names
    if inputs_numerical_column_names is None and inputs_categorical_column_names is None:
        raise ValueError('At least one input feature needs to be specified.')

    # Output regression column
    if output_regression_column_name is not None:
        if type(output_regression_column_name) is not str:
            raise TypeError('Regression output column must be specified as a string.')
        if output_regression_column_name not in data.columns:
            raise ValueError('Regression column not found in the current data frame.')

    # Output classification column
    if output_classification_column_name is not None:
        if type(output_classification_column_name) is not str:
            raise TypeError('Classification output column must be specified as a string.')
        if output_classification_column_name not in data.columns:
            raise ValueError('Classification column not found in the current data frame.')

    # Output regression and output classification columns
    if output_regression_column_name is None and output_classification_column_name is None:
        raise ValueError('Either the regression output or the classification output need to the provided.')
    if output_regression_column_name is not None and output_classification_column_name is not None:
        raise ValueError('Just one task can be completed per function call. Leave output_regression or '
                         'output_classification as None.')

    target_column = (output_classification_column_name if output_classification_column_name is not None
                     else output_regression_column_name)
    if target_column in (inputs_numerical_column_names or []) + (inputs_categorical_column_names or []):
        raise ValueError('The target column cannot also be an input feature.')

    # Learning strategy
    if learning_strategy not in ('from_scratch', 'cumulative'):
        raise ValueError('Unrecognized learning strategy.')
    else:
        if source_column_name is not None and learning_strategy == 'cumulative':
            raise ValueError('Cumulative learning can only be applied to temporal batches.')

    # Model type
    if model_type not in _MODEL_TYPES:
        raise ValueError(f'Unrecognized model type. Supported values are {_MODEL_TYPES}.')

    if (model_type == 'support_vector_machine' and output_classification_column_name is not None
            and isinstance(model_params, dict) and 'probability' in model_params):
        raise ValueError('SVC probability is managed by training-only calibration.')

    return _resolve_model_params(model_type, model_params, max_iter=max_iter,
                                 n_estimators=n_estimators, max_depth=max_depth)


# SPLITTING INDEXES OBTAINING
def _generate_split_indexes(*, data: DataFrame, batching_column_name: str, shuffle: bool = False,
                            stratify_column_name: Optional[str] = None) -> dict:
    """
    Generate split indexes based on a specified batching column (e.g., time, source).

    Parameters
    ----------
    data : DataFrame
        The input data containing the features and target variables.

    batching_column_name : str
        The column in the data used for creating splits (e.g., time, source).

    shuffle : bool, default=False
        If False, the last rows of each batch form the test set. If True, a random subset does.

    stratify_column_name : str, optional
        Label column used to stratify the shuffled split, when every class of the batch has at
        least two rows. Ignored when ``shuffle`` is False.

    Returns
    -------
    dict:
        A dictionary containing positional row indices for each batch.
    """

    # Splitting settings
    test_ratio = 0.2
    number_folds = 4
    random_seed = 42

    # Memory allocation
    split_indexes_map = dict()

    # Unique identifier values extraction
    identifiers = data[batching_column_name].unique().tolist()

    # Iteration over unique identifiers
    for idf in identifiers:
        if idf in split_indexes_map.keys():
            raise ValueError('Batching value collision.')

        # Memory allocation
        split_indexes_map[idf] = {'train_test': {}, 'puretrain_validation': {}}

        # Positions, not index labels: repeated DataFrame indices must not select
        # rows from other batches (or from the held-out test portion).
        batch_positions = np.flatnonzero(data[batching_column_name].to_numpy() == idf)

        # Training and test split
        # optional stratification of the shuffled split
        stratify = None
        if shuffle and stratify_column_name is not None:
            batch_labels = data[stratify_column_name].to_numpy()[batch_positions]
            _, class_counts = np.unique(batch_labels, return_counts=True)
            if class_counts.min() >= 2:
                stratify = batch_labels
        # train and test sets extraction
        indexes_batch_train, indexes_batch_test = train_test_split(
            batch_positions, test_size=test_ratio, random_state=random_seed, shuffle=shuffle,
            stratify=stratify
        )

        # Pure training and validation split
        # initialization
        fold_index = 0
        kfold_splitter = KFold(n_splits=number_folds, random_state=None, shuffle=False)
        # indexes generation
        for puretrain_indexes, validation_indexes in kfold_splitter.split(indexes_batch_train):
            # Arrangement
            # pure training set indexes
            split_indexes_map[idf]['puretrain_validation'][
                (f'kfold_{fold_index}', 'puretrain_indexes')] = puretrain_indexes
            # validation set indexes
            split_indexes_map[idf]['puretrain_validation'][
                (f'kfold_{fold_index}', 'validation_indexes')] = validation_indexes

            # Counter updating
            fold_index += 1

        # Arrangement
        split_indexes_map[idf]['train_test']['train_indexes'] = indexes_batch_train
        split_indexes_map[idf]['train_test']['test_indexes'] = indexes_batch_test

    # Output
    return split_indexes_map


# PERFORMANCE METRICS CALCULATION
# Single-label pre-saturation classification metrics
def _get_presaturation_classification_metrics(*, label_true: ndarray, label_scores: ndarray,
                                              index2class_map: dict) -> dict:
    """
    Calculate classification metrics (before saturation, based on probabilities).

    Parameters
    ----------
    label_true : np.ndarray
        The true class labels.

    label_scores : np.ndarray
        The predicted class probabilities.

    index2class_map : Dict[int, str]
        Mapping from class indices to class labels.

    Returns
    -------
    Dict[str, float]
        A dictionary containing classification metrics based on predicted probabilities.

    """

    # Memory allocation
    metrics = dict()

    # Metrics calculation
    # memory allocation
    auc_classes: list = []  # area under curve per class
    pr_auc_classes: list = []

    # Catch warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # single-class
        for index, class_ in index2class_map.items():
            # class identifier generation
            class_idf = str(class_).upper()
            # binarization and extraction of scores per class
            if len(label_true.shape) == 1:
                label_true_class = label_true == index
            else:  # one-hot encoding
                label_true_class = label_true[:, index]
            label_true_class = label_true_class.astype(int)
            label_scores_class = label_scores[:, index]
            # area under curve per class calculation
            try:
                auc_class = skmet.roc_auc_score(label_true_class, label_scores_class)
                pr_auc_class = skmet.average_precision_score(label_true_class, label_scores_class)
            except Exception:
                auc_class = 0
                pr_auc_class = 0
                # print('Problem calculating area under curve.')
            # arrangement
            auc_classes.append(auc_class)
            pr_auc_classes.append(pr_auc_class)
            metrics['AUC_' + class_idf] = auc_class
            metrics['PR-AUC_'+ class_idf] = pr_auc_class

        # multi-class
        # area under curve
        metrics['AUC_MACRO'] = sum(auc_classes) / len(auc_classes)
        metrics['PR-AUC_MACRO'] = sum(pr_auc_classes) / len(pr_auc_classes)
        # cross-entropy loss
        try:
            metrics['LOGLOSS'] = skmet.log_loss(label_true, label_scores, labels=list(index2class_map.keys()))
        except Exception:
            metrics['LOGLOSS'] = 1
            # print('Problem calculating logloss.')

    # Output
    return metrics


# Single-label post-saturation classification metrics
def _get_postsaturation_classification_metrics(*, label_true: ndarray, label_predicted: ndarray,
                                               index2class_map: dict) -> dict:
    """
    Calculate classification metrics after saturation (i.e., after thresholding the predicted probabilities).

    Parameters
    ----------
    label_true : np.ndarray
        The true class labels.

    label_predicted : np.ndarray
        The predicted class labels after applying a threshold (typically 0.5 for binary classification).

    index2class_map : Dict[int, str]
        Mapping from class indices to class labels.

    Returns
    -------
    Dict[str, float]
        A dictionary containing classification metrics based on predicted labels.

    """

    # Memory allocation
    metrics = dict()

    # Metrics calculation
    # Catch warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # single-class
        for index, class_ in index2class_map.items():
            # class identifier generation
            class_idf = str(class_).upper()
            # binarization
            label_true_binarized = label_true == index
            label_predicted_binarized = label_predicted == index
            # recall
            metrics['RECALL_' + class_idf] = skmet.recall_score(
                label_true_binarized, label_predicted_binarized, average='binary')
            # precision
            metrics['PRECISION_' + class_idf] = skmet.precision_score(
                label_true_binarized, label_predicted_binarized, average='binary')
            # f1_score
            metrics['F1-SCORE_' + class_idf] = skmet.f1_score(
                label_true_binarized, label_predicted_binarized, average='binary')

        # multi-class
        # accuracy
        metrics['ACCURACY'] = skmet.accuracy_score(label_true, label_predicted)
        # recall
        metrics['RECALL_MACRO'] = skmet.recall_score(label_true, label_predicted, average='macro')
        metrics['RECALL_MICRO'] = skmet.recall_score(label_true, label_predicted, average='micro')
        metrics['RECALL_WEIGHTED'] = skmet.recall_score(label_true, label_predicted, average='weighted')
        # precision
        metrics['PRECISION_MACRO'] = skmet.precision_score(label_true, label_predicted, average='macro')
        metrics['PRECISION_MICRO'] = skmet.precision_score(label_true, label_predicted, average='micro')
        metrics['PRECISION_WEIGHTED'] = skmet.precision_score(label_true, label_predicted, average='weighted')
        # f1-score
        metrics['F1-SCORE_MACRO'] = skmet.f1_score(label_true, label_predicted, average='macro')
        metrics['F1-SCORE_MICRO'] = skmet.f1_score(label_true, label_predicted, average='micro')
        metrics['F1-SCORE_WEIGHTED'] = skmet.f1_score(label_true, label_predicted, average='weighted')

    # Output
    return metrics


# Regression metrics
def _get_regression_metrics(*, y_true: ndarray, y_pred: ndarray) -> dict:
    """
    Calculate regression metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE)
    and R-squared (R2) score.

    Parameters
    ----------
    y_true : ndarray
        The true target values.

    y_pred : np.ndarray
        The predicted values from the model.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the calculated regression metrics.

    """

    # Memory allocation
    metrics = dict()

    # Metrics calculation
    # mean absolute error
    metrics['MEAN_ABSOLUTE_ERROR'] = skmet.mean_absolute_error(y_true=y_true, y_pred=y_pred)
    # mean squared error
    metrics['MEAN_SQUARED_ERROR'] = skmet.mean_squared_error(y_true=y_true, y_pred=y_pred)
    # root mean squared error
    metrics['ROOT_MEAN_SQUARED_ERROR'] = sqrt(metrics['MEAN_SQUARED_ERROR'])
    # R2
    metrics['R_SQUARED'] = skmet.r2_score(y_true=y_true, y_pred=y_pred)

    # Output
    return metrics
