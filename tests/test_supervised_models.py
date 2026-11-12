"""Run with: python3 -m unittest discover -s tests -p 'test_supervised_models.py'."""

import importlib.util
from pathlib import Path
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning


# Load the supervised module directly so this test does not depend on optional
# plotting modules imported by dashi's package-level __init__.
_path = Path(__file__).resolve().parents[1] / 'dashi/supervised_characterization/estimate_models.py'
_spec = importlib.util.spec_from_file_location('supervised_models', _path)
models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(models)


class SupervisedModelTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'site': ['a'] * 15 + ['b'] * 15,
            'date': list(pd.date_range('2020-01-01', periods=15, freq='D')) +
                    list(pd.date_range('2021-01-01', periods=15, freq='D')),
            'numeric': np.tile(np.arange(15, dtype=float), 2),
            'category': ['seen', 'other'] * 6 + ['test_only'] * 3 +
                        ['seen', 'other'] * 6 + ['another_test_only'] * 3,
            'class': ['yes', 'no'] * 7 + ['yes'] + ['no', 'yes'] * 7 + ['no'],
            'value': np.tile(np.arange(15, dtype=float), 2),
        })

    def estimate(self, data=None, *, model_type='linear', classification=True, temporal=False, **kwargs):
        arguments = dict(
            data=self.data if data is None else data,
            inputs_numerical_column_names=['numeric'],
            inputs_categorical_column_names=['category'],
            model_type=model_type,
            **({'date_column_name': 'date', 'period': 'year', 'learning_strategy': 'cumulative'}
               if temporal else {'source_column_name': 'site'}),
            **({'output_classification_column_name': 'class'} if classification else
               {'output_regression_column_name': 'value'}),
            **kwargs,
        )
        with patch.object(models, 'tqdm', side_effect=lambda iterable, **options: iterable):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                return models.estimate_multibatch_models(**arguments)

    def test_all_families_support_both_tasks_and_existing_metric_contract(self):
        for family in models._MODEL_TYPES:
            options = {'max_iter': 5} if family == 'histogram_gradient_boosting' else (
                {'n_estimators': 5} if family == 'random_forest' else (
                    {'model_params': {'hidden_layer_sizes': (4,), 'max_iter': 40}}
                    if family == 'multilayer_perceptron' else {}))
            for classification in (True, False):
                with self.subTest(family=family, classification=classification):
                    result = self.estimate(model_type=family, classification=classification, **options)
                    self.assertEqual(len(result), 4)
                    self.assertEqual(set(result), {
                        ((train,), test, 'test') for train in ('a', 'b') for test in ('a', 'b')
                    })
                    metric = 'LOGLOSS' if classification else 'MEAN_ABSOLUTE_ERROR'
                    self.assertTrue(all(np.isfinite(values[metric]) for values in result.values()))

    def test_temporal_cumulative_uses_training_portions_only(self):
        seen = []
        original_fit = models.Pipeline.fit

        def record_fit(pipeline, X, y, **kwargs):
            seen.append((pipeline, X['date'].to_numpy().copy(),
                         X['category'].tolist()))
            return original_fit(pipeline, X, y, **kwargs)

        with patch.object(models.Pipeline, 'fit', record_fit):
            result = self.estimate(temporal=True)
        self.assertEqual(len(result), 4)
        self.assertIn((('2020', '2021'), '2020', 'test'), result)
        self.assertEqual(len(seen), 2)
        self.assertEqual(len(seen[0][1]), 12)
        self.assertEqual(len(seen[1][1]), 24)
        self.assertEqual(set(seen[0][2]), {'seen', 'other'})
        self.assertEqual(set(seen[1][2]), {'seen', 'other'})
        np.testing.assert_array_equal(np.unique(seen[0][1]), ['2020'])
        self.assertIsNot(seen[0][0], seen[1][0])

    def test_train_only_scaling_and_category_vocabulary(self):
        fitted = []
        original_fit = models.Pipeline.fit

        def record_fit(pipeline, X, y, **kwargs):
            result = original_fit(pipeline, X, y, **kwargs)
            fitted.append(pipeline)
            return result

        data = self.data.copy()
        data.loc[data['site'] == 'b', 'numeric'] += 10000
        data.loc[12:14, 'numeric'] = 1e9
        with patch.object(models.Pipeline, 'fit', record_fit):
            self.estimate(data=data)
        first = fitted[0].named_steps['preprocess']
        self.assertEqual(first.named_transformers_['numerical'].center_[0], 5.5)
        self.assertEqual(set(first.named_transformers_['categorical'].categories_[0]),
                         {'seen', 'other'})
        self.assertEqual(first.transform(data.iloc[[12]]).shape[1], 3)
        np.testing.assert_array_equal(first.transform(data.iloc[[12]])[0, :2], [0, 0])

    def test_duplicate_index_labels_do_not_mix_train_and_test_rows(self):
        data = self.data.copy()
        data.index = [0] * len(data)
        split = models._generate_split_indexes(data=data, batching_column_name='site')
        for site in ('a', 'b'):
            train = split[site]['train_test']['train_indexes']
            test = split[site]['train_test']['test_indexes']
            self.assertEqual(len(train), 12)
            self.assertEqual(len(test), 3)
            self.assertFalse(set(train) & set(test))
        result = self.estimate(data=data, classification=False)
        self.assertEqual(len(result), 4)

    def test_missing_training_class_has_zero_padded_probability(self):
        data = self.data.copy()
        data.loc[15:26, 'class'] = 'no'
        result = self.estimate(data=data, model_type='random_forest', n_estimators=5)
        self.assertIn('AUC_YES', result[(('b',), 'a', 'test')])
        self.assertTrue(np.isfinite(result[(('b',), 'a', 'test')]['LOGLOSS']))

    def test_invalid_parameters_and_target_as_feature_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'probability is managed'):
            self.estimate(model_type='support_vector_machine',
                          model_params={'probability': False})
        with self.assertRaisesRegex(ValueError, 'Conflicting values for max_depth'):
            self.estimate(model_type='random_forest', max_depth=2,
                          model_params={'max_depth': 3})
        with self.assertRaisesRegex(ValueError, 'Conflicting values for max_iter'):
            self.estimate(model_type='histogram_gradient_boosting', max_iter=5,
                          model_params={'max_iter': 6})
        with self.assertRaisesRegex(ValueError, 'HGB batch settings'):
            self.estimate(model_type='histogram_gradient_boosting',
                          model_params={'categorical_features': None})
        with self.assertRaisesRegex(ValueError, 'Invalid parameter'):
            self.estimate(model_type='random_forest', model_params={'not_a_parameter': 3})
        with self.assertRaisesRegex(ValueError, 'target column'):
            models.estimate_multibatch_models(
                data=self.data, source_column_name='site',
                inputs_numerical_column_names=['value'], output_regression_column_name='value',
            )

    def test_legacy_and_model_params_produce_identical_existing_models(self):
        configurations = (
            ('random_forest', {'n_estimators': 5, 'max_depth': 3}),
            ('histogram_gradient_boosting', {'max_iter': 5}),
        )
        for family, parameters in configurations:
            for classification in (True, False):
                with self.subTest(family=family, classification=classification):
                    legacy = self.estimate(model_type=family, classification=classification,
                                           **parameters)
                    unified = self.estimate(model_type=family, classification=classification,
                                            model_params=parameters)
                    self.assertEqual(legacy, unified)
                    self.assertEqual(
                        self.estimate(model_type=family, classification=classification,
                                      model_params=parameters, **parameters), legacy,
                    )

    def test_effective_legacy_defaults_and_new_hgb_parameter(self):
        defaults = dict(max_iter=None, n_estimators=None, max_depth=None)
        self.assertEqual(models._resolve_model_params('random_forest', None, **defaults),
                         {'n_estimators': 450, 'max_depth': 9})
        self.assertEqual(models._resolve_model_params('histogram_gradient_boosting', None, **defaults),
                         {'max_iter': 100})
        self.assertEqual(models._resolve_model_params('multilayer_perceptron', None, **defaults), {})
        result = self.estimate(model_type='histogram_gradient_boosting', classification=False,
                               model_params={'max_iter': 5, 'max_depth': 2})
        self.assertEqual(len(result), 4)


if __name__ == '__main__':
    unittest.main()
