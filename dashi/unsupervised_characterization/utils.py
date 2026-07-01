import warnings
from dataclasses import dataclass
from typing import Optional, List, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import prince
from pandas.api.types import is_datetime64_any_dtype
from plotly.colors import sample_colorscale
from scipy.linalg import eigh
from scipy.stats import gaussian_kde
from sklearn.decomposition import TruncatedSVD

from dashi._constants import (VALID_FLOAT_TYPE, VALID_CATEGORICAL_TYPE, VALID_INTEGER_TYPE, VALID_STRING_TYPE,
                              VALID_CONVERSION_STRING_TYPE, VALID_PLOT_MODES, VALID_COLOR_PALETTES,
                              VALID_SORTING_METHODS, MISSING_VALUE)


class SklearnSVDAdapter:
    """
    Adapter class to make sklearn.decomposition.TruncatedSVD compatible
    with the API of the 'prince' library (PCA/MCA/FAMD).
    """

    def __init__(self, n_components: int, random_state: Optional[int] = None, **kwargs):
        # Swallow unsupported kwargs to prevent crashes, or log them if needed
        # SVD doesn't typically need scaling parameters like 'rescale_with_mean'
        self.n_components = n_components
        self.model = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.eigenvalues_summary = None  # Will be populated after fit

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits SVD and returns a DataFrame with index alignment, matching prince behavior.
        """
        # SVD works best on sparse inputs, but accepts dense DataFrames too
        matrix_transformed = self.model.fit_transform(X)

        # Calculate summary statistics immediately after fit
        self._compute_summary()

        return pd.DataFrame(
            matrix_transformed,
            index=X.index,
            columns=[i for i in range(self.n_components)]
        )

    def _compute_summary(self):
        """Generates a summary object similar to prince's eigenvalues_summary."""
        explained_var = self.model.explained_variance_ratio_
        singular_vals = self.model.singular_values_

        # Create a DataFrame summary to mimic prince's rich reporting
        self.eigenvalues_summary = pd.DataFrame({
            'singular_value': singular_vals,
            'explained_variance_ratio': explained_var,
            'cumulative_variance': np.cumsum(explained_var)
        })

def _estimate_absolute_frequencies(data, varclass, support, numeric_smoothing=False):
    """
    Estimates the absolute frequencies of data, which will be the counts_map in the final DataTemporalMap object
    """
    data = np.array(data)
    if varclass == VALID_STRING_TYPE:
        value_counts = pd.Series(data).value_counts()
        map_data = value_counts.reindex(support, fill_value=0).values

    elif varclass in VALID_FLOAT_TYPE:
        if np.all(np.isnan(data)):
            map_data =  np.full(len(support), 0, dtype=float)
        else:
            if not numeric_smoothing:
                if len(support) == 1:
                    dx_last = 1.0
                else:
                    dx_last = support[-1] - support[-2]
                hist_support = np.append(support, support[-1] + dx_last)
                data = data[(data >= min(hist_support)) & (data < max(hist_support))]
                map_data, _ = np.histogram(data, bins=hist_support)
            else:
                x = data[~np.isnan(data)]
                n_non_nan = x.size

                if n_non_nan < 4:
                    warnings.warn(
                        "Estimating a 1D KDE with < 4 data points can be inaccurate "
                        "(see Silverman, 1986, ch. 4.5.2). You may consider disabling numeric_smoothing to obtain "
                        "the histogram without KDE.")

                if n_non_nan == 0:
                    map_data = np.full(len(support), np.nan, dtype=float)

                else:
                    # Handle the case where all data points are identical
                    if np.ptp(x) == 0:
                        ndata = 1 if n_non_nan < 2 else n_non_nan
                        idx = int(np.argmin(np.abs(support - x[0])))
                        out = np.zeros_like(support, dtype=float)
                        out[idx] = float(ndata)
                        map_data = out

                    else:
                        ndata = 1 if n_non_nan < 2 else n_non_nan
                        if n_non_nan < 2:
                            x = np.repeat(x, 2)

                        kde = gaussian_kde(x)
                        y = kde(support)
                        denom = y.sum()

                        if np.isfinite(denom) and denom > 0:
                            map_data = (y / denom) * ndata
                        else:
                            map_data = np.zeros_like(y, dtype=float)

    elif varclass in VALID_INTEGER_TYPE:
        if np.all(np.isnan(data)):
            map_data = np.array([np.nan] * len(support))
        else:
            if all(isinstance(item, pd.Timestamp) for item in support):
                support_int = [item.value for item in support]
                support = support_int
            hist_support = np.append(support, support[-1] + (support[-1] - support[-2]))
            data = data[(data >= min(hist_support)) & (data < max(hist_support))]
            bin_edges = hist_support
            map_data, _ = np.histogram(data, bins=bin_edges)

    else:
        raise ValueError(f'data class {varclass} not valid for distribution estimation.')

    return map_data

def _create_supports(data, supports, columns_types, number_of_columns, numeric_variables_bins, verbose):
    supports_to_fill = {column: None for column in data.columns}
    supports_to_estimate_columns = data.columns.to_series()

    if supports is not None:
        for column_index, column in enumerate(supports):
            if column in supports_to_fill:
                supports_to_fill[column] = supports[column]
                supports_to_estimate_columns = supports_to_estimate_columns.drop(column)
                error_in_support = False

                if supports[column].dtypes == VALID_CATEGORICAL_TYPE:
                    error_in_support = (
                            not supports[column].dtype.name == VALID_CATEGORICAL_TYPE
                            and not supports[column].dtype.name == VALID_STRING_TYPE
                    )
                elif is_datetime64_any_dtype(supports[column]):
                    error_in_support = not is_datetime64_any_dtype(supports[column])
                elif supports[column].dtypes in VALID_INTEGER_TYPE:
                    error_in_support = supports[column].dtype.name not in VALID_INTEGER_TYPE
                elif supports[column].dtypes in VALID_FLOAT_TYPE:
                    error_in_support = supports[column].dtype.name not in VALID_FLOAT_TYPE

                if error_in_support:
                    raise ValueError(
                        f'The provided support for variable {column} does not match with its variable type')

    supports = supports_to_fill

    if any(supports_to_estimate_columns):
        if verbose:
            print('Estimating supports from data')

        all_na = data.loc[:, supports_to_estimate_columns].apply(lambda x: x.isnull().all())

        # Exclude from the analysis those variables with no finite values, if any
        if any(all_na):
            if verbose:
                print(
                    f'Removing variables with no finite values: {", ".join(data.columns[all_na].astype(str))}')
            warnings.warn(
                f'Removing variables with no finite values: {", ".join(data.columns[all_na].astype(str))}')

            data = data.loc[:, ~all_na]
            number_of_columns = len(data.columns)
            supports = {column_name: data_type for column_name, data_type in supports.items() if
                        not all_na[column_name]}

            data_types, columns_types = _get_types(data, verbose=False)

    categorical_mask = columns_types['categorical'] & supports_to_estimate_columns.notna()
    if categorical_mask.any():
        for name, col in data.loc[:, categorical_mask].items():
            if col.isna().any():
                # Add the category only if it's not already there
                if MISSING_VALUE not in col.cat.categories:
                    col = col.cat.add_categories([MISSING_VALUE])
                # Fill NaNs with the new category
                data[name] = col.fillna(MISSING_VALUE)

        # Extract levels and assign them to supports
        selected_columns = data.loc[:, categorical_mask]
        levels = selected_columns.apply(lambda col: col.cat.categories)
        supports.update(
            {
                column: levels[column]
                for column
                in data.columns[categorical_mask]
            }
        )
    float_mask = columns_types['float'] & supports_to_estimate_columns.notna()
    if float_mask.any():
        minimums = data.loc[:, float_mask].apply(np.nanmin, axis=0)
        maximums = data.loc[:, float_mask].apply(np.nanmax, axis=0)
        supports.update(
            {
                column: np.linspace(minimum, maximum, numeric_variables_bins).tolist()
                for column, minimum, maximum
                in zip(data.columns[float_mask], minimums,
                       maximums)
            }
        )
        is_constant = minimums == maximums
        if is_constant.any():
            # Get the names of the constant columns
            constant_cols = is_constant[is_constant].index

            # Update only the constant columns
            supports.update({
                col: [supports[col][0]]
                for col in constant_cols
            })

    int_mask = columns_types['integer'] & supports_to_estimate_columns.notna()
    if int_mask.any():
        minimums = data.loc[:, int_mask].apply(np.nanmin, axis=0)
        maximums = data.loc[:, int_mask].apply(np.nanmax, axis=0)
        supports.update(
            {
                column: np.linspace(minimum, maximum, numeric_variables_bins).tolist()
                for column, minimum, maximum
                in
                zip(data.columns[int_mask], minimums,
                    maximums)
            }
        )
        is_constant = minimums == maximums
        if is_constant.any():
            constant_cols = is_constant[is_constant].index
            supports.update({
                col: [supports[col][0]]
                for col in constant_cols
            })

    str_mask = columns_types['string'] & supports_to_estimate_columns.notna()
    if str_mask.any():
        supports.update(
            {
                column: data[column].unique().tolist()
                for column
                in data.columns[str_mask]
            }
        )

    date_mask = columns_types['date'] & supports_to_estimate_columns.notna()
    if date_mask.any():
        minimums = data.loc[:, date_mask].apply(np.nanmin, axis=0)
        maximums = data.loc[:, date_mask].apply(np.nanmax, axis=0)
        supports.update(
            {
                column: pd.date_range(minimum, maximum, periods=numeric_variables_bins).tolist()
                for column, minimum, maximum
                in zip(data.columns[date_mask], minimums,
                       maximums)
            }
        )
        is_constant = minimums == maximums
        if is_constant.any():
            constant_cols = is_constant[is_constant].index
            supports.update({
                col: [supports[col][0]]
                for col in constant_cols
            })

    # Convert factor variables to characters, as used by the xts Objects
    if np.any(columns_types['categorical']):
        converted_columns = data.loc[:, columns_types['categorical']].astype(
            VALID_CONVERSION_STRING_TYPE)
        data = data.assign(**converted_columns)

    # Exclude from the analysis those variables with a single value, if any
    support_lengths = [len(supports[column]) for column in data.columns]
    support_singles_indexes = np.array(support_lengths) < 2
    if np.any(support_singles_indexes):
        if verbose:
            print(
                f'Removing variables with less than two distinct values in their supports: {", ".join(data.columns[support_singles_indexes].astype(str))}')
        print(
            f'The following variable/s have less than two distinct values in their supports and were excluded from the analysis: {", ".join(data.columns[support_singles_indexes].astype(str))}')
        data = data.loc[:, ~support_singles_indexes]
        supports = {
            column: supports[column]
            for column
            in data.columns
        }
        number_of_columns = len(data.columns)

    if number_of_columns == 0:
        raise ValueError('Zero remaining variables to be analyzed.')

    return data, supports

def _get_types(data, verbose=False):
    data_types = data.dtypes

    float_columns = data_types.astype(str).isin(VALID_FLOAT_TYPE)
    integer_columns = data_types.astype(str).isin(VALID_INTEGER_TYPE)
    string_columns = data_types == VALID_STRING_TYPE
    date_columns = data_types.apply(lambda dtype: is_datetime64_any_dtype(dtype))
    categorical_columns = data_types == VALID_CATEGORICAL_TYPE

    columns_by_type = {
        "float": float_columns,
        "integer": integer_columns,
        "string": string_columns,
        "date": date_columns,
        "categorical": categorical_columns
    }

    if verbose:
        if any(columns_by_type["float"]):
            print(f'Number of float columns: {sum(columns_by_type["float"])}')
        if any(columns_by_type["integer"]):
            print(f'Number of integer columns: {sum(columns_by_type["integer"])}')
        if any(columns_by_type["string"]):
            print(f'Number of string columns: {sum(columns_by_type["string"])}')
        if any(columns_by_type["date"]):
            print(f'Number of date columns: {sum(columns_by_type["date"])}')
        if any(columns_by_type["categorical"]):
            print(f'Number of categorical columns: {sum(columns_by_type["categorical"])}')

    return data_types, columns_by_type

def _validate_plot_args(
        mode,
        color_palette,
        absolute,
        log_transform,
        start_value,
        sorting_method,
        valid_sorting_methods=None,
) -> None:
    if mode is not None and mode not in VALID_PLOT_MODES:
        raise ValueError(f'mode must be one of the defined in {VALID_PLOT_MODES}')

    if color_palette is not None and color_palette not in VALID_COLOR_PALETTES:
        raise ValueError(f'color_palette must be one of the defined in {VALID_COLOR_PALETTES}')

    if not isinstance(absolute, bool):
        raise ValueError('absolute must be a logical value')

    if not isinstance(log_transform, bool):
        raise ValueError('log_transform must be a logical value')

    if not isinstance(start_value, int) and start_value < 1:
        raise ValueError('start_value must be greater or equal than 1')

    if valid_sorting_methods is None:
        valid_sorting_methods = VALID_SORTING_METHODS

    if sorting_method not in valid_sorting_methods:
        raise ValueError(f'sorting_method must be one of the defined in {valid_sorting_methods}')

def _sort_support_and_map(
        support,
        data_map,
        variable_type,
        sorting_method
):
    if variable_type in [VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE]:
        if sorting_method in ['frequency']:
            support_order = np.argsort(np.nansum(data_map, axis=0))[::-1]
        else:
            support_order = np.argsort(support)

        support = support[support_order]

        # Resort temporal map by support_order
        data_map = np.array([row[support_order] for row in data_map])

        any_supp_na = pd.isnull(support)
        if any_supp_na.any():
            support[any_supp_na] = '<NA>'

    return support, data_map


def _sort_support_and_map_by_reference(
        support,
        data_map,
        variable_type,
        reference_support
):
    if variable_type in [VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE]:
        reference_order = {value: index for index, value in enumerate(reference_support)}
        support_order = sorted(
            range(len(support)),
            key=lambda index: reference_order.get(support[index], len(reference_order) + index)
        )

        support = support[support_order]

        # Resort temporal/source map by support_order
        data_map = np.array([row[support_order] for row in data_map])

        any_supp_na = pd.isnull(support)
        if any_supp_na.any():
            support[any_supp_na] = '<NA>'

    return support, data_map


def _get_joint_frequency_support(
        supports,
        data_maps,
        variable_type
):
    if variable_type not in [VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE]:
        return None

    frequencies = dict()
    first_seen = dict()
    next_index = 0
    for support, data_map in zip(supports, data_maps):
        support_frequencies = np.sum(data_map, axis=0)
        for support_value, frequency in zip(support, support_frequencies):
            if support_value not in frequencies:
                frequencies[support_value] = 0
                first_seen[support_value] = next_index
                next_index += 1

            frequencies[support_value] += frequency

    return np.array(sorted(frequencies, key=lambda value: (-frequencies[value], first_seen[value])))


def _get_counts_array(
        data_map,
        start_value,
        end_value,
        log_transform,
        temporal=False
):
    if log_transform:
        data_map = np.log(data_map + 1e-8)

    counts_subarray = [row[start_value: end_value] for row in data_map]
    if temporal:
        counts_subarray = list(zip(*counts_subarray))

    return counts_subarray

def _create_heatmap_figure(
        data_map,
        x,
        y,
        z,
        color_palette,
        font,
        x_axis,
        title,
        absolute
):
    figure = go.Figure(
        data=go.Heatmap(
            x=x,
            y=y,
            z=z,
            colorscale=color_palette,
            reversescale=True,
            colorbar=dict(
                tickfont=dict(
                    color='black'
                )
            )
        )
    )

    y_axis = dict(
        title=data_map.variable_name,
        titlefont=font,
        automargin=True,
        tickfont={'color': 'black'},
        ticks='outside',
        tickcolor='black'
    )

    figure.update_xaxes(x_axis)

    figure.update_layout(yaxis=y_axis,
                         autosize=True,
                         paper_bgcolor='white',
                         plot_bgcolor='white'
                         )

    if data_map.variable_type in [VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE]:
        figure.update_layout(yaxis_type='category')

    if title is not None:
        figure.update_layout(title={'text': title,
                                    'font': {'color': 'black'
                                             }
                                    })
    else:
        title = 'Probability distribution data temporal heatmap'
        if absolute:
            title = 'Absolute frequencies data temporal heatmap'
        figure.update_layout(title={'text': title,
                                    'font': {'color': 'black'
                                             }
                                    }
                             )
    return figure

def _create_series_figure(
        data_map,
        x,
        y,
        name,
        absolute,
        x_axis,
        font,
        title,
        _range,
        **kwargs
):
    figure = go.Figure()

    for i in _range:
        trace = go.Scatter(
            x=x,
            y=y[i],
            mode='lines',
            name=str(name[i])
        )
        figure.add_trace(trace)

    if absolute:
        y_axis_title = 'Absolute frequency'
    else:
        y_axis_title = 'Relative frequency'

    y_axis = dict(
        title=y_axis_title,
        titlefont=font,
        automargin=True,
        tickfont={'color': 'black'},
        ticks='outside',
        tickcolor='black'
    )

    figure.update_xaxes(
        x_axis
    )

    figure.update_layout(
        autosize=True,
        yaxis=y_axis,
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend_title_text=data_map.variable_name if 'temporal' in kwargs else 'Source'
    )

    if title is not None:
        figure.update_layout(title={'text': title,
                                    'font': {'color': 'black'
                                             }
                                    }
                             )
    else:
        figure.update_layout(title={'text': 'Evolution of ' + data_map.variable_name,
                                    'font': {'color': 'black'
                                             }
                                    }
                             )
    return figure

def _js_divergence(p, q, epsilon=1e-10):
    """
    Computes the Jensen-Shannon (JS) divergence between two probability distributions.

    The Jensen-Shannon divergence is a symmetric and smoothed version of the Kullback-Leibler (KL) divergence
    and measures the similarity between two probability distributions. Unlike the KL divergence, which is asymmetric,
    the JS divergence is symmetric and bounded, making it a more stable measure for comparing distributions.

    The JS divergence is calculated as:
    JS(p || q) = 0.5 * (KL(p || m) + KL(q || m))
    where m = 0.5 * (p + q) is the average distribution, and KL(p || m) is the Kullback-Leibler divergence between
    distribution `p` and `m`.
    """
    p = np.asarray(p)
    q = np.asarray(q)

    p = np.where(p < epsilon, epsilon, p)
    q = np.where(q < epsilon, epsilon, q)

    m = 0.5 * (p + q)

    kl_p_m = np.where(p != 0, p * np.log2(p / m), 0)
    kl_q_m = np.where(q != 0, q * np.log2(q / m), 0)

    result = 0.5 * (np.nansum(kl_p_m) + np.nansum(kl_q_m))

    return result

def _cmdscale(d, k=2, eig=False, add=False, x_ret=False):
    """
    Performs Classical Multidimensional Scaling (MDS) on a distance matrix to reduce the dimensionality
    of the data, while preserving pairwise distances as much as possible.
    """
    # Check for NA values (Not Applicable in numpy, but we can check for NaN)
    if np.isnan(d).any():
        raise ValueError("NA values not allowed in 'd'")

    list_ = eig or add or x_ret

    if not list_:
        if eig:
            print("Warning: eig=TRUE is disregarded when list_=FALSE")
        if x_ret:
            print("Warning: x_ret=TRUE is disregarded when list_=FALSE")

    if not isinstance(d, np.ndarray) or len(d.shape) != 2 or d.shape[0] != d.shape[1]:
        if add:
            d = np.array(d)
        x = np.array(d ** 2, dtype=np.double)
        n = x.shape[0]
        if n != x.shape[1]:
            raise ValueError("distances must be result of 'dist' or a square matrix")
        rn = np.arange(n)
    else:
        n = d.shape[0]
        rn = np.arange(n)
        x = np.zeros((n, n))
        if add:
            d0 = x.copy()
        triu_indices = np.triu_indices_from(x, 1)
        x[triu_indices] = d[triu_indices] ** 2
        x += x.T
        if add:
            d0[triu_indices] = d[triu_indices]
            d = d0 + d0.T

    if not isinstance(n, int) or n > 46340:
        raise ValueError("invalid value of 'n'")

    if k > n - 1 or k < 1:
        raise ValueError("'k' must be in {1, 2, ..  n - 1}")

    # Double centering
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H.dot(x).dot(H)

    if add:
        i2 = n + np.arange(n)
        Z = np.zeros((2 * n, 2 * n))
        Z[np.arange(n), i2] = -1
        Z[i2, np.arange(n)] = -x
        Z[i2, i2] = 2 * d
        e = np.linalg.eigvals(Z)
        add_c = np.max(np.real(e))
        x = np.zeros((n, n), dtype=np.double)
        non_diag = np.triu_indices_from(d, 1)
        x[non_diag] = (d[non_diag] + add_c) ** 2
        x = -0.5 * H.dot(x).dot(H)

    e_vals, e_vecs = eigh(B)
    idx = np.argsort(e_vals)[::-1]
    e_vals = e_vals[idx]
    e_vecs = e_vecs[:, idx]

    ev = e_vals[:k]
    evec = e_vecs[:, :k]
    k1 = np.sum(ev > 0)

    if k1 < k:
        print(f"Warning: only {k1} of the first {k} eigenvalues are > 0")
        evec = evec[:, ev > 0]
        ev = ev[ev > 0]

    points = evec * np.sqrt(ev)

    if list_:
        evalus = e_vals
        return {
            'points': points,
            'eig': evalus if eig else None,
            'x': B if x_ret else None,
            'ac': add_c if add else 0,
            'GOF': np.sum(ev) / np.array([np.sum(np.abs(evalus)), np.sum(np.maximum(evalus, 0))])
        }
    else:
        return points

@dataclass
class BaseMultiVariateMap:
    multivariate_probability_map: Optional[np.array] = None
    multivariate_counts_map: Optional[np.array] = None
    multivariate_support: Optional[np.array] = None

    def check_multivariate(self) -> Union[List[str], bool]:
        errors = []
        if self.multivariate_probability_map is not None and self.multivariate_counts_map is not None:
            if len(self.multivariate_probability_map) != len(self.multivariate_counts_map) or \
               any(len(prob_row) != len(count_row)
                   for prob_row, count_row in zip(self.multivariate_probability_map, self.multivariate_counts_map)):
                errors.append("The dimensions of multivariate_probability_map and multivariate_counts_map do not match.")
        if self.multivariate_support is not None and self.multivariate_probability_map is not None:
            prob_shape = np.shape(self.multivariate_probability_map[0])
            if len(self.multivariate_support) != len(prob_shape) or \
               any(len(support_axis) != axis_length
                   for support_axis, axis_length in zip(self.multivariate_support, prob_shape)):
                errors.append("The length of multivariate_support must match the columns of multivariate_probability_map.")
        if self.multivariate_support is not None and self.multivariate_counts_map is not None:
            counts_shape = np.shape(self.multivariate_counts_map[0])
            if len(self.multivariate_support) != len(counts_shape) or \
               any(len(support_axis) != axis_length
                   for support_axis, axis_length in zip(self.multivariate_support, counts_shape)):
                errors.append("The length of multivariate_support must match the columns of multivariate_counts_map.")
        return errors if errors else True

def _perform_dimensionality_reduction(
        data: pd.DataFrame,
        dim_reduction: str,
        n_components: int,
        verbose: bool = True,
        **reduction_kwargs) -> pd.DataFrame:
    """
        Factory function to perform dimensionality reduction using various strategies.
        Supports PCA, MCA, FAMD (via prince) and SVD (via sklearn adapter).
    """

    reduction_strategies = {
        'PCA': prince.PCA,
        'MCA': prince.MCA,
        'FAMD': prince.FAMD,
        'SVD': SklearnSVDAdapter
    }

    MethodClass = reduction_strategies[dim_reduction]
    algo_kwargs = reduction_kwargs.copy()

    if 'scale' in reduction_kwargs:
        if dim_reduction == 'PCA':
            scale_value = algo_kwargs.pop('scale')
            algo_kwargs['rescale_with_mean'] = scale_value
            algo_kwargs['rescale_with_std'] = scale_value
        else:
            algo_kwargs.pop('scale')

    try:
        reduction_method = MethodClass(n_components=n_components, random_state=112, **algo_kwargs)
    except TypeError as e:
        raise TypeError(f"Error initializing {dim_reduction}. Check valid arguments. Details: {e}")
    reduced_data = reduction_method.fit_transform(data)

    if verbose:
        print(f"--- {dim_reduction} Summary ---")
        # All wrappers/classes are guaranteed to have this attribute now
        if hasattr(reduction_method, 'eigenvalues_summary'):
            print(reduction_method.eigenvalues_summary)
        else:
            print("No summary available for this method.")

    return reduced_data

def _scatter_plot(reduced_data: pd.DataFrame, dim_reduction: str, verbose: bool, **kwargs) -> None:
    warnings.filterwarnings('ignore', category=FutureWarning)
    if verbose:
        print(f'Plotting {dim_reduction} 2D Scatter Plot')

    color_col = kwargs.get('color_column', None)
    if color_col is not None and not pd.api.types.is_numeric_dtype(reduced_data[color_col]):
        categories = sorted(reduced_data[color_col].dropna().unique().tolist())
        stops = [i / (len(categories) - 1) if len(categories) > 1 else 0.5 for i in range(len(categories))]
        palette = sample_colorscale('Viridis', stops)
        color_map = {cat: col for cat, col in zip(categories, palette)}

    fig = px.scatter(
        reduced_data,
        x=0,
        y=1,
        title=f'{dim_reduction} Scatter Plot',
        template='plotly_white',
        opacity=0.7,
        color=color_col,
        category_orders={color_col: categories},
        color_discrete_map=color_map
    )

    fig.update_layout(
        title={
            'text': f'{dim_reduction} Scatter Plot',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 25}
        },
        xaxis_title={
            'text': f'PC1',
            'font': {'size': 18}
        },
        yaxis_title={
            'text': f'PC2',
            'font': {'size': 18}
        }

    )
    fig.show()
    warnings.filterwarnings('default', category=FutureWarning)

def _compute_kde(data_subset, xmin, xmax, kde_resolution):
    """
    Performs a Gaussian Kernel Density Estimation (KDE) on a subset of the original data to estimate
    the probability density function over a specified range, using a grid resolution determined by
    kde_resolution.
    """
    data_subset = np.array(data_subset)
    kde = gaussian_kde(data_subset.T, bw_method='silverman')  # Transpose for data compatibility
    grid = [np.linspace(start, stop, kde_resolution) for start, stop in zip(xmin, xmax)]
    mesh = np.meshgrid(*grid, indexing='ij')
    positions = np.vstack([m.ravel() for m in mesh])
    kde_values = kde(positions).reshape([kde_resolution] * len(xmin))
    return kde_values


def _normalize_kde(kde_values):
    """
    Normalizes the results of the Kernel Density Estimation (KDE) values so that the total area under the
    estimated probability density function equals 1. This ensures that the KDE represents a valid probability
    distribution.
    """
    kde_values = np.maximum(kde_values, 0)  # Set negative values to 0
    return kde_values / np.sum(kde_values)  # Normalize

def _marginalize_multivariate_map(multivariate_map, supports, dimensions) -> list:
    probability_map_list: list = []

    if dimensions == 1:
        probability_map_dim1 = pd.DataFrame(multivariate_map, columns=supports[0])
        probability_map_list.append(probability_map_dim1)

    elif dimensions == 2:
        probability_map_dim1 = pd.DataFrame([np.sum(dim1, axis=1) for dim1 in multivariate_map],
                                            columns=supports[0])
        probability_map_dim2 = pd.DataFrame([np.sum(dim2, axis=0) for dim2 in multivariate_map],
                                            columns=supports[1])
        probability_map_list.extend([probability_map_dim1, probability_map_dim2])

    elif dimensions == 3:
        probability_map_dim1 = pd.DataFrame([np.sum(dim1, axis=(2, 1)) for dim1 in multivariate_map],
                                            columns=supports[0])
        probability_map_dim2 = pd.DataFrame([np.sum(dim2, axis=(2, 0)) for dim2 in multivariate_map],
                                            columns=supports[1])
        probability_map_dim3 = pd.DataFrame([np.sum(dim3, axis=(0, 1)) for dim3 in multivariate_map],
                                            columns=supports[2])
        probability_map_list.extend([probability_map_dim1, probability_map_dim2, probability_map_dim3])

    return probability_map_list

def _date_to_numeric(data, columns_by_type, verbose):
    # Convert dates to numbers
    if any(columns_by_type['date']):
        date_cols = data.columns[columns_by_type['date']]
        for col in date_cols:
            data[col] = pd.to_datetime(data[col], errors='coerce').astype('int64')
        if verbose:
            print('Converting date columns to numeric for distribution analysis')
    return data
