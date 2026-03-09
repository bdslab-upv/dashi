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
    HistGradientBoostingRegressor
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder, OrdinalEncoder
from tqdm.auto import tqdm

__all__ = ['estimate_multibatch_models']


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
    model_type: Optional[str] = 'histogram_gradient_boosting'
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
        Defines the model family to use: either 'random_forest' or 'histogram_gradient_boosting'.
        When 'random_forest' is selected, categorical features are one-hot encoded and a
        ``RandomForestClassifier``/``RandomForestRegressor`` is used.
        When 'histogram_gradient_boosting' is selected, categorical features are ordinal encoded
        (with native categorical support) and a
        ``HistGradientBoostingClassifier``/``HistGradientBoostingRegressor`` is used.

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

    _check_inputs(
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
    )

    # Modeling settings
    random_seed = 42
    use_hgb = model_type == "histogram_gradient_boosting"

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
    split_indexes = _generate_split_indexes(data=df_work, batching_column_name=batching_column_name)
    # Extract batch identifiers
    batch_identifiers = list(split_indexes.keys())

    cat_cols = inputs_categorical_column_names or []
    num_cols = inputs_numerical_column_names or []

    if len(cat_cols) + len(num_cols) == 0:
        raise ValueError("No input features provided.")

    # ---------- Preprocessing helpers per model type ----------

    def _fit_transform_features(df_train: pd.DataFrame):
        """
        Fit encoder/scaler on training data and return (X_train, encoder, scaler, final_input_features).
        """
        final_input_features = []
        encoder = None
        scaler = None
        df_out = df_train.copy()

        # Categorical encoding
        if cat_cols:
            if use_hgb:
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                    dtype=np.float32,
                )
                df_out[cat_cols] = encoder.fit_transform(df_out[cat_cols])
                final_input_features.extend(cat_cols)
            else:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                encoder.fit(df_out[cat_cols])
                encoded_cols = encoder.get_feature_names_out(cat_cols).tolist()
                encoded_df = pd.DataFrame(
                    encoder.transform(df_out[cat_cols]),
                    columns=encoded_cols,
                    index=df_out.index,
                )
                df_out = df_out.drop(columns=cat_cols)
                df_out = pd.concat([df_out, encoded_df], axis=1)
                final_input_features.extend(encoded_cols)

        # Numerical scaling
        if num_cols:
            scaler = RobustScaler()
            df_out[num_cols] = scaler.fit_transform(df_out[num_cols])
            final_input_features.extend(num_cols)

        X = df_out[final_input_features].to_numpy(dtype=np.float32, copy=True)
        return X, encoder, scaler, final_input_features

    def _transform_features(df_test: pd.DataFrame, encoder, scaler, final_input_features):
        """
        Transform test data using already-fitted encoder/scaler.
        """
        df_out = df_test.copy()

        if encoder is not None:
            if use_hgb:
                df_out[cat_cols] = encoder.transform(df_out[cat_cols])
            else:
                encoded_cols = encoder.get_feature_names_out(cat_cols).tolist()
                encoded_df = pd.DataFrame(
                    encoder.transform(df_out[cat_cols]),
                    columns=encoded_cols,
                    index=df_out.index,
                )
                df_out = df_out.drop(columns=cat_cols)
                df_out = pd.concat([df_out, encoded_df], axis=1)

        if scaler is not None:
            df_out[num_cols] = scaler.transform(df_out[num_cols])

        X = df_out[final_input_features].to_numpy(dtype=np.float32, copy=True)
        return X

    # ---------- Main loop ----------

    metrics: Dict[Tuple, Dict[str, float]] = {}

    train_loop = tqdm(
        enumerate(batch_identifiers),
        total=len(batch_identifiers),
        colour="#32CD32",
        desc="Learning and testing over experiences",
    )

    for idx, batch_idf_train in train_loop:

        # --- Resolve training indices (label-based) ---
        if learning_strategy == "from_scratch":
            current_train_batch_ids = (batch_idf_train,)
            train_label_indices = split_indexes[batch_idf_train]["train_test"]["train_indexes"]
        elif learning_strategy == "cumulative":
            current_train_batch_ids = tuple(batch_identifiers[i] for i in range(0, idx + 1))
            train_label_indices = np.concatenate(
                [split_indexes[bid]["train_test"]["train_indexes"] for bid in current_train_batch_ids]
            )
        else:
            raise ValueError("Unrecognized learning strategy.")

        train_sub = df_work.loc[train_label_indices]

        # --- Preprocessing (fit on train) ---
        X_train, encoder, scaler, final_input_features = _fit_transform_features(train_sub)

        # --- Build model ---
        # HGB: mark categorical feature positions for native categorical splits
        categorical_features = list(range(len(cat_cols))) if (cat_cols and use_hgb) else None

        if output_regression_column_name is not None:
            y_train = train_sub[output_regression_column_name].to_numpy(copy=False)
            if use_hgb:
                model = HistGradientBoostingRegressor(
                    max_iter=100,
                    random_state=random_seed,
                    early_stopping=True,
                    scoring="loss",
                    categorical_features=categorical_features,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=450,
                    max_depth=9,
                    random_state=random_seed,
                    n_jobs=-1,
                )
        else:
            y_train = train_sub[output_classification_column_name].to_numpy(copy=False)
            if use_hgb:
                model = HistGradientBoostingClassifier(
                    max_iter=100,
                    random_state=random_seed,
                    class_weight="balanced",
                    early_stopping=True,
                    scoring="loss",
                    categorical_features=categorical_features,
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=450,
                    max_depth=9,
                    random_state=random_seed,
                    class_weight="balanced",
                    n_jobs=-1,
                )

        model.fit(X_train, y_train)

        # Free training buffers
        del train_sub, X_train, y_train
        gc.collect()

        # --- Testing loop ---
        for batch_idf_test in batch_identifiers:
            test_key = (current_train_batch_ids, batch_idf_test, "test")

            test_label_indices = split_indexes[batch_idf_test]["train_test"]["test_indexes"]
            test_sub = df_work.loc[test_label_indices]

            X_test = _transform_features(test_sub, encoder, scaler, final_input_features)

            if output_regression_column_name is not None:
                y_test = test_sub[output_regression_column_name].to_numpy(copy=False)
                y_pred = model.predict(X_test)
                metrics[test_key] = _get_regression_metrics(y_true=y_test, y_pred=y_pred)
            else:
                y_test = test_sub[output_classification_column_name].to_numpy(copy=False)
                probs_hat = model.predict_proba(X_test)
                labels_hat = model.predict(X_test)

                index2index_map = dict(enumerate(model.classes_))
                index2class_map_batch = {
                    i: index2class_map[index2index_map[i]] for i in index2index_map
                }

                metrics_pre = _get_presaturation_classification_metrics(
                    label_true=y_test,
                    label_scores=probs_hat,
                    index2class_map=index2class_map_batch,
                )
                metrics_post = _get_postsaturation_classification_metrics(
                    label_true=y_test,
                    label_predicted=labels_hat,
                    index2class_map=index2class_map_batch,
                )
                metrics[test_key] = {**metrics_pre, **metrics_post}

            del test_sub, X_test
            gc.collect()

        # Free model + preprocessors
        del model, scaler, encoder
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
        model_type: Optional[str]
) -> None:
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
        Defines the model family: 'random_forest' or 'histogram_gradient_boosting'.

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

    # Learning strategy
    if learning_strategy not in ('from_scratch', 'cumulative'):
        raise ValueError('Unrecognized learning strategy.')
    else:
        if source_column_name is not None and learning_strategy == 'cumulative':
            raise ValueError('Cumulative learning can only be applied to temporal batches.')

    # Model type
    if model_type not in ('random_forest', 'histogram_gradient_boosting'):
        raise ValueError("Unrecognized model type. Supported values are 'random_forest' and "
                         "'histogram_gradient_boosting'.")


# SPLITTING INDEXES OBTAINING
def _generate_split_indexes(*, data: DataFrame, batching_column_name: str) -> dict:
    """
    Generate split indexes based on a specified batching column (e.g., time, source).

    Parameters
    ----------
    data : DataFrame
        The input data containing the features and target variables.

    batching_column_name : str
        The column in the data used for creating splits (e.g., time, source).

    Returns
    -------
    dict:
        A dictionary containing the split indexes for each batch. The keys are batch identifiers,
        and the values are dictionaries with 'train' and 'test' indexes.
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

        # Data batch extraction
        data_batch = data[data[batching_column_name] == idf]

        # Training and test split
        # train and test sets extraction
        data_batch_train, data_batch_test = train_test_split(
            data_batch, test_size=test_ratio, random_state=random_seed, shuffle=False
        )
        # indexes extraction
        #   training
        indexes_batch_train = data_batch_train.index.to_numpy()
        #   test
        indexes_batch_test = data_batch_test.index.to_numpy()

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
            metrics['LOGLOSS'] = skmet.log_loss(label_true, label_scores)
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
