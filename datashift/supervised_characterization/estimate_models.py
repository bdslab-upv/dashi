"""
DESCRIPTION: main function for estimating models over multiple temporal or multi-source batches.
AUTHOR: Pablo Ferri-Borredà
DATE: 17/10/24
"""

# MODULES IMPORT
from typing import List, Dict, Optional

import sklearn.metrics as skmet
from dateutil.parser import parse as parse_date
from numpy import ndarray, sqrt
from pandas import DataFrame, concat, get_dummies
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from tqdm import tqdm


# FUNCTION DEFINITION
def estimate_multi_batch_models(*, data: DataFrame, inputs_numerical_columns: List[str],
                                inputs_categorical_columns: List[str], output_regression_column: Optional[str] = None,
                                output_classification_column: Optional[str] = None, date_column: Optional[str] = None,
                                batching_period: Optional[str] = None,
                                source_column: Optional[str] = None) -> Dict[str, float]:
    # Input checking
    _check_inputs(
        data=data, inputs_numerical_columns=inputs_numerical_columns,
        inputs_categorical_columns=inputs_categorical_columns, output_regression_column=output_regression_column,
        output_classification_column=output_classification_column, date_column=date_column,
        batching_period=batching_period, source_column=source_column
    )

    # Memory allocation
    metrics = {}

    # Modeling settings
    number_trees = 450
    maximum_depth = 9
    random_seed = 42

    # Label encoding
    if output_classification_column is not None:
        # label encoder initialization
        label_encoder = LabelEncoder()
        # label encoding
        data[output_classification_column] = label_encoder.fit_transform(data[output_classification_column])
        # index to class map derivation
        index2class_map = dict(enumerate(label_encoder.classes_))

    # Batching column extraction
    # multi-source analysis
    if date_column is None and source_column is not None:
        batching_column = source_column
    # temporal analysis
    if date_column is not None and source_column is None:
        # date parsing
        data[date_column] = data[date_column].apply(lambda date_string: parse_date(date_string))
        # batching period adjusting
        if batching_period == 'MONTH':
            data[date_column] = data[date_column].apply(lambda date_: date_.strftime("%B %Y"))
        elif batching_period == 'YEAR':
            data[date_column] = data[date_column].apply(lambda date_: date_.strftime("%Y"))
        else:
            raise ValueError("Current supported batching periods are 'MONTH' and 'YEAR'.")
        # batching column assignation
        batching_column = date_column
    else:
        raise ValueError('This casuistry has not been implemented yet.')

    # One-hot encoding of categorical features
    inputs_categorical_columns_ = inputs_categorical_columns.copy()
    for cat_col in inputs_categorical_columns_:
        data_encoded = get_dummies(data[cat_col], prefix=cat_col, prefix_sep='-', drop_first=False)
        data = concat([data, data_encoded], axis=1)
        data = data.drop(columns=[cat_col])

        inputs_categorical_columns.remove(cat_col)
        inputs_categorical_columns.extend(list(data_encoded.columns))

    # Split indexes generation
    split_indexes = _generate_split_indexes(data=data, batching_column=batching_column)

    # Batch identifiers extraction
    batch_identifiers = split_indexes.keys()

    # Combinations generation
    # memory allocation
    combinations = []
    # filling
    for batch_idf_train in batch_identifiers:
        combinations.append((batch_idf_train, batch_idf_train, 'train',))
        for batch_idf_test in batch_identifiers:
            combinations.append((batch_idf_train, batch_idf_test, 'test'))

    # Preprocessing, training and evaluation
    for combination in tqdm(combinations, total=len(combinations), colour='#32CD32',
                            desc='Learning and testing over experiences', position=0, leave=True):
        # Identifiers extraction
        # batch identifier
        batch_idf = combination[1]
        # data set
        data_set = combination[2]

        # Metrics dictionary checking
        if combination in metrics.keys():
            raise ValueError('Batch already visited.')

        # Extraction of the train and test data
        data_batch = data.loc[split_indexes[batch_idf]['train_test'][f'{data_set}_indexes']]

        # Continuous features preprocessing
        #   data selection
        data_batch_cont = data_batch[inputs_numerical_columns]
        #   scaler initialization and training (if required)
        if data_set == 'train':
            # scaler initialization
            robust_scaler = RobustScaler()
            # scaler training
            robust_scaler.fit(data_batch_cont)
        #   scaling
        data_batch[inputs_numerical_columns] = robust_scaler.transform(data_batch_cont)

        # Inputs extraction
        inputs_batch = concat(
            [data_batch[inputs_numerical_columns], data_batch[inputs_categorical_columns]], axis=1
        )

        # Regression pipeline
        if output_regression_column is not None:
            # Outputs extraction
            outputs_batch = data_batch[output_regression_column]

            # Model initialization and training
            if data_set == 'train':
                # initialization
                model = RandomForestRegressor(
                    n_estimators=number_trees, max_depth=maximum_depth, random_state=random_seed
                )
                # training
                model.fit(inputs_batch, outputs_batch)

            # Inference
            y_pred = model.predict(inputs_batch)

            # Regression metrics calculation
            metrics_regression = _get_regression_metrics(y_true=outputs_batch, y_pred=y_pred)

            # Regression metrics arrangement
            metrics[combination] = metrics_regression

        # Classification pipeline
        elif output_classification_column is not None:
            # Outputs extraction
            outputs_batch = data_batch[output_classification_column]

            # Model initialization and training
            if data_set == 'train':
                # initialization
                model = RandomForestClassifier(
                    n_estimators=number_trees, max_depth=maximum_depth, random_state=random_seed,
                    class_weight='balanced'
                )
                # training
                model.fit(inputs_batch, outputs_batch)

            # Inference
            # index correspondance extraction
            index2index_map = dict(enumerate(model.classes_))
            index2class_map_batch = {idx: index2class_map[index2index_map[idx]] for idx in index2index_map.keys()}
            # raw probabilities extraction
            probs_hat = model.predict_proba(inputs_batch)
            # saturated values extraction
            labels_hat = model.predict(inputs_batch)

            # Metrics calculation
            # pre-saturation metrics calculation
            metrics_presatur = _get_presaturation_classification_metrics(
                label_true=outputs_batch, label_scores=probs_hat, index2class_map=index2class_map_batch
            )
            # post-saturation metrics calculation
            metrics_postsatur = _get_postsaturation_classification_metrics(
                label_true=outputs_batch, label_predicted=labels_hat, index2class_map=index2class_map_batch
            )

            # Arrangement
            # metrics combination
            metrics_combined = {**metrics_presatur, **metrics_postsatur}
            # metrics arrangement
            metrics[combination] = metrics_combined

        # Unconsidered casuistry
        else:
            raise ValueError('This casuistry is not allowed.')

    # Output
    return metrics


# INPUTS CHECKING
def _check_inputs(*, data: DataFrame, inputs_numerical_columns: List[str],
                  inputs_categorical_columns: List[str], output_regression_column: Optional[str] = None,
                  output_classification_column: Optional[str] = None, date_column: Optional[str] = None,
                  batching_period: Optional[str] = None,
                  source_column: Optional[str] = None) -> None:
    # Data
    if type(data) is not DataFrame:
        raise TypeError('Data must be encapsulated into a Data frame object.')
    else:
        if data.isnull().values.any():
            raise ValueError('Missing data is present in your data frame object. '
                             'Please, process them before calling this function.')

    # Date column
    if date_column is not None:
        if type(date_column) is not str:
            raise TypeError('Date column must be specified as a string.')
        if date_column not in data.columns:
            raise ValueError('Date column not found in the current data frame.')
        # batching period
        if batching_period is None:
            raise ValueError("A batching period needs to be specified: either 'MONTH' or 'YEAR'.")
        else:
            if batching_period not in ('MONTH', 'YEAR'):
                raise ValueError("Current supported batching periods are 'MONTH' and 'YEAR'.")

    # Source column
    if source_column is not None:
        if type(source_column) is not str:
            raise TypeError('Source column must be specified as a string.')
        if source_column not in data.columns:
            raise ValueError('Source column not found in the current data frame.')

    # Date and source column
    if date_column is None and source_column is None:
        raise ValueError('Either the date column or the source column needs to the provided.')
    if date_column is not None and source_column is not None:
        raise ValueError('Just one batching column can be considered (date or source but not both simultaneously).')

    # Inputs numerical columns
    if type(inputs_numerical_columns) is not list:
        raise TypeError('Numerical inputs columns need to be encapsulated in a list.')
    for inp_num_col in inputs_numerical_columns:
        if type(inp_num_col) is not str:
            raise TypeError('Numerical input column must be specified as a string.')
        else:
            if inp_num_col not in data.columns:
                raise ValueError('Numerical input column not found in the current data frame.')

    # Inputs categorical columns
    if type(inputs_categorical_columns) is not list:
        raise TypeError('Categorical inputs columns need to be encapsulated in a list.')
    for inp_cat_col in inputs_categorical_columns:
        if type(inp_cat_col) is not str:
            raise TypeError('Categorical input column must be specified as a string.')
        else:
            if inp_cat_col not in data.columns:
                raise ValueError('Categorical input column not found in the current data frame.')

    # Output regression column
    if output_regression_column is not None:
        if type(output_regression_column) is not str:
            raise TypeError('Regression output column must be specified as a string.')
        if output_regression_column not in data.columns:
            raise ValueError('Regression column not found in the current data frame.')

    # Output classification column
    if output_classification_column is not None:
        if type(output_classification_column) is not str:
            raise TypeError('Classification output column must be specified as a string.')
        if output_classification_column not in data.columns:
            raise ValueError('Classification column not found in the current data frame.')

    # Output regression and output classification columns
    if output_regression_column is None and output_classification_column is None:
        raise ValueError('Either the regression output or the classification output need to the provided.')
    if output_regression_column is not None and output_classification_column is not None:
        raise ValueError('Just one task can be completed per function call. Leave output_regression or '
                         'output_classification as None.')


# SPLITTING INDEXES OBTAINING
def _generate_split_indexes(*, data: DataFrame, batching_column: str) -> dict:
    # Splitting settings
    test_ratio = 0.2
    number_folds = 4
    random_seed = 42

    # Memory allocation
    split_indexes_map = dict()

    # Unique identifier values extraction
    identifiers = data[batching_column].unique().tolist()

    # Iteration over unique identifiers
    for idf in identifiers:
        if idf in split_indexes_map.keys():
            raise ValueError('Batching value collision.')

        # Memory allocation
        split_indexes_map[idf] = {'train_test': {}, 'puretrain_validation': {}}

        # Data batch extraction
        data_batch = data[data[batching_column] == idf]

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
    # Memory allocation
    metrics = dict()

    # Metrics calculation
    # memory allocation
    auc_classes = []  # area under curve per class

    # single-class
    for index, class_ in index2class_map.items():
        # class identifier generation
        class_idf = class_.upper()
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
        except:
            auc_class = 0
            print('Problem calculating area under curve.')
        # arrangement
        auc_classes.append(auc_class)
        metrics['AUC_' + class_idf] = auc_class

    # multi-class
    # area under curve
    metrics['AUC_MACRO'] = sum(auc_classes) / len(auc_classes)
    # cross-entropy loss
    try:
        metrics['LOGLOSS'] = skmet.log_loss(label_true, label_scores)
    except:
        metrics['LOGLOSS'] = 1
        print('Problem calculating logloss.')

    # Output
    return metrics


# Single-label post-saturation classification metrics
def _get_postsaturation_classification_metrics(*, label_true: ndarray, label_predicted: ndarray,
                                               index2class_map: dict) -> dict:
    # Memory allocation
    metrics = dict()

    # Metrics calculation
    # single-class
    for index, class_ in index2class_map.items():
        # class identifier generation
        class_idf = class_.upper()
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
    metrics['PRECISION_MICRO'] = skmet.recall_score(label_true, label_predicted, average='micro')
    metrics['PRECISION_WEIGHTED'] = skmet.recall_score(label_true, label_predicted, average='weighted')
    # f1-score
    metrics['F1-SCORE_MACRO'] = skmet.f1_score(label_true, label_predicted, average='macro')
    metrics['F1-SCORE_MICRO'] = skmet.f1_score(label_true, label_predicted, average='micro')
    metrics['F1-SCORE_WEIGHTED'] = skmet.f1_score(label_true, label_predicted, average='weighted')

    # Output
    return metrics


# Regression metrics
def _get_regression_metrics(*, y_true: ndarray, y_pred: ndarray) -> dict:
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
