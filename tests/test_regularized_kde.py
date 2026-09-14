"""Run with: python3 -m unittest discover -s tests -p 'test_regularized_kde.py'."""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, multivariate_normal

from dashi.unsupervised_characterization import utils
from dashi.unsupervised_characterization.data_source_map import data_source_map as source
from dashi.unsupervised_characterization.data_temporal_map import data_temporal_map as temporal


class RegularizedKDETests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(123)

    def test_unregularized_and_zero_ridge_match_scipy(self):
        x = self.rng.normal(size=(25, 2))
        grid = np.linspace(-3, 3, 17)
        positions = np.array(np.meshgrid(grid, grid, indexing='ij')).reshape(2, -1)
        expected = gaussian_kde(x.T, bw_method='silverman')(positions).reshape(17, 17)
        actual = utils._compute_kde(x, [-3, -3], [3, 3], 17)
        np.testing.assert_allclose(actual, expected)
        regularized = utils._compute_kde(x, [-3, -3], [3, 3], 17, np.zeros((2, 2)))
        np.testing.assert_allclose(utils._normalize_kde(regularized), utils._normalize_kde(expected))

    def test_positive_ridge_matches_independent_gaussian_mixture(self):
        x = self.rng.normal(size=(11, 3))
        additive = np.diag([0.1, 0.2, 0.3])
        factor = (len(x) * 5 / 4) ** (-1 / 7)
        covariance = factor ** 2 * np.cov(x, rowvar=False) + additive
        grid = np.linspace(-4, 4, 9)
        points = np.array(np.meshgrid(grid, grid, grid, indexing='ij')).reshape(3, -1).T
        expected = sum(multivariate_normal.pdf(points, mean=row, cov=covariance) for row in x)
        actual = utils._compute_kde(x, [-4] * 3, [4] * 3, 9, additive)
        np.testing.assert_allclose(utils._normalize_kde(actual).ravel(), expected / expected.sum())

    def test_constant_collinear_singleton_and_one_dimension(self):
        for x in (np.ones((6, 2)), np.array([[0., 0.], [1., 1.]]),
                  np.array([[2., -1.]]), np.ones((4, 1))):
            d = x.shape[1]
            kde = utils._compute_kde(x, [-4] * d, [4] * d, 21, np.eye(d) * 0.1)
            probabilities = utils._normalize_kde(kde)
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertAlmostEqual(probabilities.sum(), 1)

    def test_validation_and_pooled_degeneracy(self):
        for strength in (-1, np.inf, np.nan, '0.05', [0.05], 1j):
            with self.assertRaises(ValueError):
                utils._validate_regularization(strength)
        frame = pd.DataFrame({'x': [1., 1.], 'y': [0., 1.], 'site': ['a', 'b']})
        with self.assertRaisesRegex(ValueError, 'pooled variance'):
            utils._prepare_regularized_kde(frame, ['site'], 0.05, 20)
        with self.assertRaisesRegex(ValueError, 'kde_resolution'):
            utils._prepare_regularized_kde(frame, ['site'], 0.05, 1)

    def test_reference_tail_bounds_and_grid_warning(self):
        frame = pd.DataFrame({'x': [-1., 1., 0.], 'y': [0., 0., 2.], 'site': ['a', 'a', 'b']})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            additive, support = utils._prepare_regularized_kde(frame, ['site'], 0.05, 10)
        expected = 0.05 ** 2 * frame[['x', 'y']].var().to_numpy()
        np.testing.assert_allclose(np.diag(additive), expected)
        self.assertTrue(any('kde_resolution' in str(item.message) for item in caught))
        self.assertTrue(any('rank-deficient' in str(item.message) for item in caught))
        self.assertTrue(np.all(support['xmin'] < frame[['x', 'y']].min().to_numpy()))
        self.assertTrue(np.all(support['xmax'] > frame[['x', 'y']].max().to_numpy()))

    def test_grid_refinement_converges_on_gaussian_second_moment(self):
        errors = []
        for resolution in (9, 33, 65):
            grid = np.linspace(-6, 6, resolution)
            kde = utils._compute_kde(np.array([[0.]]), [-6], [6], resolution, np.eye(1))
            second_moment = np.sum(utils._normalize_kde(kde) * grid ** 2)
            errors.append(abs(second_moment - 1))
        self.assertLess(errors[-1], errors[0])
        self.assertLess(errors[-1], 1e-6)

    def test_source_error_and_counts_with_sparse_and_empty_groups(self):
        x = np.vstack([self.rng.normal(size=(7, 2)), np.zeros((4, 2)), [[2., 1.]]])
        frame = pd.DataFrame(x, columns=['x', 'y'])
        frame['site'] = ['c'] * 7 + ['b'] * 4 + ['a']
        info = {'sources': np.array(['a', 'b', 'c', 'empty']),
                'value_counts': frame.site.value_counts(sort=False), 'source_column_name': 'site'}
        with self.assertRaisesRegex(np.linalg.LinAlgError, "source .*'b'.*regularization_strength"):
            source._generate_multivariate_dsm(frame, info, False, 2, 12)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            additive, support = utils._prepare_regularized_kde(frame, ['site'], 0.2, 30)
        result = source._generate_multivariate_dsm(frame, info, False, 2, 30, support, additive)
        for index, count in enumerate([1, 4, 7]):
            np.testing.assert_allclose(result.counts_map[index], np.round(result.probability_map[index] * count))
            self.assertAlmostEqual(result.probability_map[index].sum(), 1)
        self.assertTrue(np.isnan(result.probability_map[-1]).all())

    def test_temporal_skip_keeps_counts_aligned(self):
        dates = pd.date_range('2020-01-01', periods=4, freq='MS')
        x = np.vstack([self.rng.normal(size=(8, 2)), np.zeros((5, 2)), self.rng.normal(size=(6, 2))])
        frame = pd.DataFrame(x, columns=['x', 'y'])
        frame['date'] = [dates[2]] * 8 + [dates[1]] * 5 + [dates[0]] * 6
        info = {'unique_dates': dates, 'value_counts': frame.date.value_counts(sort=False),
                'date_column_name': 'date', 'period': 'month'}
        with self.assertWarnsRegex(RuntimeWarning, 'regularization_strength'):
            result = temporal._generate_multivariate_dtm(frame, info, False, 2, 15, context="label 'test', ")
        for index, count in [(0, 6), (2, 8)]:
            np.testing.assert_allclose(result.counts_map[index], np.round(result.probability_map[index] * count))
        self.assertTrue(np.isnan(result.probability_map[[1, 3]]).all())

    def test_all_public_entrypoints_and_shared_conditional_support(self):
        # Identity reduction isolates group handling and lets exact degeneracy persist.
        frame = pd.DataFrame(self.rng.normal(size=(16, 2)), columns=['x', 'y'])
        frame.loc[:3, ['x', 'y']] = 0.
        frame['site'] = ['a'] * 4 + ['b'] * 4 + ['a'] * 4 + ['b'] * 4
        frame['date'] = pd.to_datetime(['2020-01-01'] * 4 + ['2020-02-01'] * 4
                                        + ['2020-01-01'] * 4 + ['2020-02-01'] * 4)
        frame['label'] = ['first'] * 8 + ['second'] * 8
        frame = frame.astype({'site': object, 'label': object})
        with patch.object(source, '_perform_dimensionality_reduction', side_effect=lambda data, **kw: data.copy()), \
                patch.object(temporal, '_perform_dimensionality_reduction', side_effect=lambda data, **kw: data.copy()), \
                warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            options = dict(regularization_strength=0.2, kde_resolution=20)
            maps = [source.estimate_multivariate_data_source_map(frame.drop(columns=['date', 'label']), 'site', **options),
                    temporal.estimate_multivariate_data_temporal_map(frame.drop(columns=['site', 'label']), 'date', **options)]
            for module, function, group, drop in (
                    (source, source.estimate_conditional_data_source_map, 'site', 'date'),
                    (temporal, temporal.estimate_conditional_data_temporal_map, 'date', 'site')):
                with patch.object(module, '_prepare_regularized_kde', wraps=utils._prepare_regularized_kde) as prepare:
                    conditional = function(frame.drop(columns=[drop]), group, 'label', **options)
                    self.assertEqual(prepare.call_count, 1)
                first, second = conditional.values()
                np.testing.assert_allclose(first.multivariate_support, second.multivariate_support)
                maps.extend(conditional.values())
            for result in maps:
                self.assertTrue(np.isfinite(result.probability_map).all())
                np.testing.assert_allclose(result.probability_map.sum(axis=1), 1)

    def test_public_pca_integration(self):
        frame = pd.DataFrame(self.rng.normal(size=(24, 3)), columns=['x', 'y', 'z'])
        frame.loc[:5, ['x', 'y', 'z']] = 0.
        frame['site'] = ['constant'] * 6 + ['variable'] * 18
        frame = frame.astype({'site': object})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = source.estimate_multivariate_data_source_map(
                frame, 'site', regularization_strength=0.2, kde_resolution=20)
        np.testing.assert_allclose(result.probability_map.sum(axis=1), 1)


if __name__ == '__main__':
    unittest.main()
