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
Data Source Map plotting main functions and classes
"""
from typing import Optional, Dict, List, Union

import numpy as np
import plotly.colors
import plotly.graph_objs as go
import plotly.subplots as sp

from dashi.unsupervised_characterization.data_source_map.data_source_map import DataSourceMap, MultiVariateDataSourceMap
from dashi.unsupervised_characterization.utils import (_validate_plot_args, _sort_support_and_map, _get_counts_array,
                                                       _marginalize_multivariate_map, _create_series_figure,
                                                       _get_joint_frequency_support,
                                                       _sort_support_and_map_by_reference)

__all__ = [
    'plot_univariate_data_source_map',
    'plot_conditional_univariate_data_source_map',
    'plot_multivariate_data_source_map',
    'plot_conditional_data_source_map'
]


def _prepare_univariate_source_plot_data(
        data_source_map: DataSourceMap,
        absolute: bool,
        log_transform: bool,
        start_value: Optional[int],
        end_value: Optional[int],
        sorting_method: str,
        reference_support=None
):
    if absolute:
        source_map = data_source_map.counts_map
    else:
        source_map = data_source_map.probability_map

    sources = data_source_map.sources
    support = np.array(data_source_map.support.iloc[:, 0].tolist())
    variable_type = data_source_map.variable_type

    if reference_support is None:
        support, source_map = _sort_support_and_map(
            support=support,
            data_map=source_map,
            variable_type=variable_type,
            sorting_method=sorting_method
        )
    else:
        support, source_map = _sort_support_and_map_by_reference(
            support=support,
            data_map=source_map,
            variable_type=variable_type,
            reference_support=reference_support
        )

    if not end_value or end_value > source_map.shape[1]:
        end_value = source_map.shape[1]

    if start_value > source_map.shape[1]:
        start_value = source_map.shape[1]

    counts_subarray = _get_counts_array(
        data_map=source_map,
        start_value=start_value,
        end_value=end_value,
        log_transform=log_transform
    )

    return data_source_map, sources, support, counts_subarray, start_value, end_value


def plot_univariate_data_source_map(
        data_source_map: Union[DataSourceMap, Dict[str, DataSourceMap]],
        variable_name: Optional[str] = None,
        absolute: bool = False,
        log_transform: bool = False,
        start_value: Optional[int] = 0,
        end_value: Optional[int] = None,
        sorting_method: str = 'alphabetical',
        title: Optional[str] = None
) -> go.Figure:
    """
    Plots a Data Source heatmap or series from a DataSourceMap object.

    Parameters
    ----------
    data_source_map : DataSourceMap | Dict[str, DataSourceMap]
        The DataSourceMap object that contains data to be plotted, or a dictionary of DataSourceMap objects returned by
        estimate_univariate_data_source_map.

    variable_name : str, optional
        The variable to plot when data_source_map is a dictionary of DataSourceMap objects.

    absolute : bool
        If True, plot absolute values; otherwise, the relative probabilities are plotted. Default is False.

    log_transform : bool
        If True, applies a log transformation to the data for better visibility of small values. Default is False.

    start_value : int, optional
        The value at which to start the plot. Default is 0.

    end_value : int, optional
        The value at which to end the plot. If None, the plot extends to the last value. Default is None.

    sorting_method : str, optional
        The method by which the data will be sorted for display (e.g., 'frequency', 'alphabetical').
        Default is 'frequency'.

    title : str, optional
        The title of the plot. If None, a default title is used. Default is None.

    Returns
    -------
    Figure
        The Plotly figure object representing the plot
    """
    if type(data_source_map) == dict:
        if len(data_source_map) == 0:
            raise ValueError('data_source_map dictionary must contain at least one DataSourceMap.')

        if not all(type(value) == DataSourceMap for value in data_source_map.values()):
            raise TypeError('data_source_map dictionary values must be DataSourceMap objects.')

        if variable_name is None:
            raise ValueError('variable_name must be provided when data_source_map is a dictionary.')

        if variable_name not in data_source_map:
            raise ValueError(f'Variable {variable_name} not found in data_source_map.')

        data_source_map = data_source_map[variable_name]

    if not type(data_source_map) == DataSourceMap:
        raise TypeError('data_source_map must be an instance of DataSourceMap.')
    _validate_plot_args(
        mode=None,
        color_palette=None,
        absolute=absolute,
        log_transform=log_transform,
        start_value=start_value,
        sorting_method=sorting_method,
        valid_sorting_methods=['frequency', 'alphabetical']
    )

    data_source_map, sources, support, counts_subarray, start_value, end_value = _prepare_univariate_source_plot_data(
        data_source_map=data_source_map,
        absolute=absolute,
        log_transform=log_transform,
        start_value=start_value,
        end_value=end_value,
        sorting_method=sorting_method
    )

    font = dict(size=20, color='#7f7f7f')
    x_axis = dict(title=data_source_map.variable_name,
                  titlefont=font,
                  tickvals=support,
                  tickfont={'color': 'black'},
                  ticks='outside',
                  tickcolor='black',
                  tickangle=45
                  )

    figure = _create_series_figure(
        data_map=data_source_map,
        x=support,
        y=counts_subarray,
        name=sources,
        absolute=absolute,
        x_axis=x_axis,
        font=font,
        title=title,
        _range=range(len(data_source_map.sources))
    )
    return figure


def plot_conditional_univariate_data_source_map(
        data_source_map_dict: Dict[str, Union[DataSourceMap, Dict[str, DataSourceMap]]],
        variable_name: Optional[str] = None,
        absolute: bool = False,
        log_transform: bool = False,
        start_value: Optional[int] = 0,
        end_value: Optional[int] = None,
        sorting_method: str = 'alphabetical',
        title: Optional[str] = None
) -> go.Figure:
    """
    Plots conditional univariate Data Source Maps from a dictionary of labels.

    Parameters
    ----------
    data_source_map_dict : Dict[str, DataSourceMap | Dict[str, DataSourceMap]]
        A dictionary where keys are labels. Values can be DataSourceMap objects, or dictionaries of DataSourceMap
        objects when multiple variables were estimated per label.

    variable_name : str, optional
        The variable to plot when each label maps to a dictionary of DataSourceMap objects.

    absolute : bool
        If True, plot absolute values; otherwise, relative probabilities are plotted. Default is False.

    log_transform : bool
        If True, applies a log transformation to the data. Default is False.

    start_value : int, optional
        The value at which to start the plot. Default is 0.

    end_value : int, optional
        The value at which to end the plot. If None, the plot extends to the last value.

    sorting_method : str, optional
        The method by which the support values will be sorted for display (e.g., 'frequency', 'alphabetical',
        'joint_frequency'). The 'frequency' methods shorts each label independently, while the 'joint_frequency'
        applys the same category order across all conditional labels based on their joint frequency.
        Default is 'alphabetical'.

    title : str, optional
        The title of the plot. If None, a default title is used.

    Returns
    -------
    Figure
        The Plotly figure object representing the conditional univariate data source map.
    """
    if not type(data_source_map_dict) == dict:
        raise TypeError('data_source_map_dict must be a dictionary.')

    if len(data_source_map_dict) == 0:
        raise ValueError('data_source_map_dict must contain at least one label.')

    _validate_plot_args(
        mode=None,
        color_palette=None,
        absolute=absolute,
        log_transform=log_transform,
        start_value=start_value,
        sorting_method=sorting_method
    )

    selected_maps = dict()
    values_are_maps = [type(value) == DataSourceMap for value in data_source_map_dict.values()]
    values_are_dicts = [type(value) == dict for value in data_source_map_dict.values()]

    if all(values_are_maps):
        selected_maps = data_source_map_dict
    elif all(values_are_dicts):
        if variable_name is None:
            raise ValueError('variable_name must be provided when each label maps to multiple DataSourceMap objects.')

        for label, maps_by_variable in data_source_map_dict.items():
            if variable_name not in maps_by_variable:
                raise ValueError(f'Variable {variable_name} not found for label {label}.')

            selected_maps[label] = maps_by_variable[variable_name]
    else:
        raise TypeError('data_source_map_dict values must be all DataSourceMap objects or all dictionaries.')

    for data_source_map in selected_maps.values():
        if not type(data_source_map) == DataSourceMap:
            raise TypeError('Selected conditional maps must be DataSourceMap objects.')

    labels = list(selected_maps.keys())
    reference_support = None
    if sorting_method == 'joint_frequency':
        supports = list()
        data_maps = list()
        for data_source_map in selected_maps.values():
            supports.append(np.array(data_source_map.support.iloc[:, 0].tolist()))
            data_maps.append(data_source_map.counts_map if absolute else data_source_map.probability_map)

        reference_support = _get_joint_frequency_support(
            supports=supports,
            data_maps=data_maps,
            variable_type=next(iter(selected_maps.values())).variable_type
        )

    prepared_maps = dict()
    all_sources = []
    for label, data_source_map in selected_maps.items():
        prepared_maps[label] = _prepare_univariate_source_plot_data(
            data_source_map=data_source_map,
            absolute=absolute,
            log_transform=log_transform,
            start_value=start_value,
            end_value=end_value,
            sorting_method=sorting_method,
            reference_support=reference_support
        )

        for source in data_source_map.sources:
            if source not in all_sources:
                all_sources.append(source)

    palette = plotly.colors.qualitative.Plotly
    colors = {source: palette[i % len(palette)] for i, source in enumerate(all_sources)}

    subplot = sp.make_subplots(
        rows=len(labels),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.09
    )

    font = dict(size=20, color='#7f7f7f')
    for row, label in enumerate(labels, start=1):
        data_source_map, sources, support, counts_subarray, row_start_value, row_end_value = prepared_maps[label]
        support_slice = support[row_start_value:row_end_value]

        for source_index, source in enumerate(sources):
            subplot.add_trace(
                go.Scatter(
                    x=support_slice,
                    y=counts_subarray[source_index],
                    mode='lines',
                    name=str(source),
                    showlegend=row == 1,
                    legendgroup=str(source),
                    line=dict(color=colors[source])
                ),
                row=row,
                col=1
            )

        subplot.update_yaxes(
            title=str(label),
            titlefont=font,
            automargin=True,
            row=row,
            col=1,
            ticks='outside',
            tickcolor='black'
        )

        subplot.update_xaxes(
            tickvals=support_slice,
            tickfont={'size': 12},
            tickangle=45,
            title_text=data_source_map.variable_name if row == len(labels) else None,
            title_font=font if row == len(labels) else None,
            row=row,
            col=1,
            ticks='outside',
            tickcolor='black'
        )

    if title is None:
        title = f'{"Absolute frequencies" if absolute else "Probability distribution"} conditional data source map of {variable_name}'

    subplot.update_layout(
        autosize=True,
        height=max(450, min(380 * len(labels), 1200)),
        showlegend=True,
        legend_title_text='Source',
        template='plotly_white',
        margin=dict(t=80, r=30, b=100, l=80),
        title={'text': title, 'font': {'color': 'black'}}
    )

    return subplot


def plot_multivariate_data_source_map(
        data_source_map: MultiVariateDataSourceMap,
        absolute: bool = False
) -> go.Figure:
    """
    Plots a multivariate Data Source heatmap from a MultiVariateDataSourceMap object.

    Parameters
    ----------
    data_source_map : MultiVariateDataSourceMap
        The MultiVariateDataSourceMap object that contains multivariate data to be plotted.

    absolute : bool, optional
        If True, plot absolute values; otherwise, the relative probabilities are plotted. Default is False.

    Returns
    -------
    Figure
        The Plotly figure object representing the multivariate heatmap.
    """
    if not type(data_source_map) == MultiVariateDataSourceMap:
        raise TypeError('data_source_map must be an instance of MultiVariateDataSourceMap, obtained from the'
                        ' estimate_multivariate_data_source_map function.')

    if not isinstance(absolute, bool):
        raise TypeError('absolute must be a boolean value, indicating whether to plot absolute counts or probabilities.')

    sources = data_source_map.sources
    supports = data_source_map.multivariate_support
    dimensions = len(supports)

    # Create a color palette for the sources
    palette = plotly.colors.qualitative.Plotly
    colors = {source: palette[i % len(palette)] for i, source in enumerate(sources)}

    if absolute:
        multivariate_map = data_source_map.multivariate_counts_map
    else:
        multivariate_map = data_source_map.multivariate_probability_map

    probability_map_list = _marginalize_multivariate_map(
        multivariate_map=multivariate_map,
        supports=supports ,
        dimensions=dimensions)

    subplot = sp.make_subplots(
        rows=dimensions,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.07
    )
    font = dict(size=20, color='#7f7f7f')

    for i, source_map in enumerate(probability_map_list):
        support = np.array(source_map.columns)
        counts_subarray = [row for row in source_map.values]

        for j in range(len(sources)):
            trace = go.Scatter(
                x=support,
                y=counts_subarray[j],
                mode='lines',
                name=str(sources[j]),
                showlegend= (i == 0),
                legendgroup=sources[j],
                line=dict(color=colors[sources[j]])
            )
            subplot.add_trace(trace, row=i + 1, col=1)

        subplot.update_yaxes(
            title=f'PC {i + 1}',
            titlefont=font,
            automargin=True,
            row=i + 1,
            col=1,
            ticks='outside',
            tickcolor='black'
        )

        subplot.update_xaxes(
            tickvals=support,
            tickformat='.2f',
            tickangle=15,
            tickfont={'size': 12},
            title_font=font,
            title_text='Support' if i == dimensions - 1 else None,
            # title_font=font if i == dimensions - 1 else None,
            row=i + 1,
            col=1,
            ticks='outside',
            tickcolor='black'
        )

    subplot.update_layout(
        autosize=True,
        height=max(450, min(380 * dimensions, 1200)),
        showlegend=True,
        legend_title_text='Source',
        template='plotly_white',
        margin=dict(t=80, r=30, b=100, l=80),
        coloraxis=dict(colorscale='Spectral_r'),
        title=f'{"Absolute frequencies" if absolute else "Probability distribution"} '
              f'data source map'
    )
    return subplot

def plot_conditional_data_source_map(
        data_source_map_dict: Dict[str, MultiVariateDataSourceMap],
        absolute: bool = False
) -> List[go.Figure]:
    """
    Plots a Figure for each dimension selected in the data_temporal_map_dict. Each Figure represents the
    Data Temporal heatmap of each label in that dimension

    Parameters
    ----------
    data_source_map_dict : Dict[str, MultiVariateDataSourceMap]
        A dictionary where keys are labels (strings), and values are the corresponding
        `MultiVariateDataSourceMap` objects obtained from the 'estimate_conditional_data_source_map' function.

    absolute : bool, optional
        If True, plot absolute values; otherwise, relative probabilities are plotted. Default is False.

    Returns
    -------
    conditional_plots_list : List[Figure]
        A list of Plotly figure objects representing the conditional data source maps for each dimension.
    """

    if not isinstance(data_source_map_dict, dict) or not all(
            isinstance(value, MultiVariateDataSourceMap) for value in data_source_map_dict.values()
    ):
        raise TypeError('data_source_map_dict must be a dictionary with MultiVariateDataSourceMap instances, resultant'
                        'of the estimate_conditional_data_source_map function.')

    if len(data_source_map_dict) == 0:
        raise ValueError('data_source_map_dict must contain at least one MultiVariateDataSourceMap.')

    if not isinstance(absolute, bool):
        raise TypeError('absolute must be a boolean value, indicating whether to plot absolute counts or probabilities.')

    expected_dimensions = None
    for label, data_source_map in data_source_map_dict.items():
        if data_source_map.multivariate_support is None or len(data_source_map.multivariate_support) == 0:
            raise ValueError('Each MultiVariateDataSourceMap must contain multivariate_support.')

        dimensions = len(data_source_map.multivariate_support)
        if expected_dimensions is None:
            expected_dimensions = dimensions
        elif dimensions != expected_dimensions:
            raise ValueError('All MultiVariateDataSourceMap objects must have the same number of dimensions.')

    labels = list(data_source_map_dict.keys())
    probability_map_dict: dict = {}
    sources_dict: dict = {}

    for label, data_source_map in data_source_map_dict.items():
        sources_dict[label] = data_source_map.sources
        supports = data_source_map.multivariate_support
        dimensions = len(supports)

        if absolute:
            if dimensions == 1:
                multivariate_map = data_source_map.counts_map
            else:
                multivariate_map = data_source_map.multivariate_counts_map
        else:
            if dimensions == 1:
                multivariate_map = data_source_map.probability_map
            else:
                multivariate_map = data_source_map.multivariate_probability_map

        probability_map_list = _marginalize_multivariate_map(
            multivariate_map=multivariate_map,
            supports=supports,
            dimensions=dimensions
        )

        probability_map_dict[label] = probability_map_list

    conditional_plots_list = list()
    for dim in range(dimensions):
        subplot = sp.make_subplots(
            rows=len(labels),
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08
        )

        font = dict(size=20, color='#7f7f7f')

        for i, (label, probability_map_list) in enumerate(probability_map_dict.items()):
            sources = sources_dict[label]
            source_map = probability_map_list[dim]
            support = np.array(source_map.columns)
            counts_subarray = [row for row in source_map.values]

            # Create a color palette for the sources
            palette = plotly.colors.qualitative.Plotly
            colors = {source: palette[i % len(palette)] for i, source in enumerate(sources)}

            for j in range(len(sources)):
                trace = go.Scatter(
                    x=support,
                    y=counts_subarray[j],
                    mode='lines',
                    name=str(sources[j]),
                    showlegend= (i == 0),
                    legendgroup=sources[j],
                    line=dict(color=colors[sources[j]])
                )
                subplot.add_trace(trace, row=labels.index(label) + 1, col=1)

            subplot.update_yaxes(
                title=f'{label}',
                titlefont=font,
                automargin=True,
                row=labels.index(label) + 1,
                col=1,
                ticks='outside',
                tickcolor='black'
            )

            subplot.update_xaxes(
                tickvals=support,
                tickformat='.2f',
                tickfont={'size': 12},
                tickangle=45,
                title_text='Support' if labels.index(label) == len(labels) - 1 else None,
                title_font=font if labels.index(label) == len(labels) - 1 else None,
                row=labels.index(label) + 1,
                col=1,
                ticks='outside',
                tickcolor='black'
            )

        subplot.update_layout(
            autosize=True,
            height=max(450, min(380 * len(labels), 1200)),
            showlegend=True,
            legend_title_text='Source',
            template='plotly_white',
            margin=dict(t=80, r=30, b=100, l=80),
            coloraxis=dict(colorscale='Spectral_r'),
            title=f'{"Absolute frequencies" if absolute else "Probability distribution"} '
                  f'conditional data source map of Principal Component {dim + 1}'
        )

        conditional_plots_list.append(subplot)
    return conditional_plots_list

