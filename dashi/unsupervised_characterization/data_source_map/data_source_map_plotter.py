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
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.io as pio
from typing import Optional, Dict

from dashi._constants import VALID_SORTING_METHODS, VALID_COLOR_PALETTES, \
    VALID_PLOT_MODES, VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE
from dashi.unsupervised_characterization.data_source_map.data_source_map import DataSourceMap, MultiVariateDataSourceMap
from dashi.unsupervised_characterization.utils import (_validate_plot_args, _sort_support_and_map, _get_counts_array,
                                                       _create_heatmap_figure, _create_series_figure,
                                                       _marginalize_multivariate_map)

pio.renderers.default = "browser"

def plot_univariate_data_source_map(
        data_source_map: DataSourceMap,
        absolute: bool = False,
        log_transform: bool = False,
        start_value: Optional[int] = 0,
        end_value: Optional[int] = None,
        sorting_method: str = 'alphabetical',
        color_palette: str = 'Spectral',
        mode: str = 'heatmap',
        title: Optional[str] = None
) -> go.Figure:
    """
    Plots a Data Source heatmap or series from a DataSourceMap object.

    Parameters
    ----------
    data_source_map : DataTemporalMap
        The DataSourceMap object that contains data to be plotted.

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

    color_palette : str, optional
        The color palette to be used for the plot (e.g., 'Spectral', 'viridis', 'viridis_r', 'magma', 'magma_r).
        Default is 'Spectral'.

    mode : str, optional
        The mode of visualization (e.g., 'heatmap', 'series'). Default is 'heatmap'.

    title : str, optional
        The title of the plot. If None, a default title is used. Default is None.

    Returns
    -------
    Figure
        The Plotly figure object representing the plot
    """
    if not type(data_source_map) == DataSourceMap:
        raise TypeError('data_source_map must be an instance of DataSourceMap.')
    _validate_plot_args(
        mode=mode,
        color_palette=color_palette,
        absolute=absolute,
        log_transform=log_transform,
        start_value=start_value,
        sorting_method=sorting_method
    )

    if absolute:
        source_map = data_source_map.counts_map
    else:
        source_map = data_source_map.probability_map

    sources = data_source_map.sources
    support = np.array(data_source_map.support.iloc[:, 0].tolist())
    variable_type = data_source_map.variable_type

    support, source_map = _sort_support_and_map(
        support=support,
        data_map=source_map,
        variable_type=variable_type,
        sorting_method=sorting_method
    )

    if not end_value or end_value > source_map.shape[0]:
        end_value = source_map.shape[1]

    if start_value > source_map.shape[0]:
        start_value = source_map.shape[1]

    counts_subarray = _get_counts_array(
        data_map=source_map,
        start_value=start_value,
        end_value=end_value,
        log_transform=log_transform
    )

    font = dict(size=20, color='#7f7f7f')
    x_axis = dict(title='Source',
                  tickvals=sources[::2] if len(sources) > 2 else sources,
                  titlefont={'color': 'black'},
                  tickfont={'color': 'black'},
                  ticks='outside',
                  tickcolor='black')


    if mode == 'heatmap':
        figure = _create_heatmap_figure(
            data_map=data_source_map,
            x=sources,
            y=support[start_value:end_value],
            z=counts_subarray,
            color_palette=color_palette,
            font=font,
            x_axis=x_axis,
            title=title,
            absolute=absolute,
        )

        return figure

    elif mode == 'series':
        figure = _create_series_figure(
            data_map=data_source_map,
            x=sources,
            y=counts_subarray,
            name=support,
            absolute=absolute,
            start_value=start_value,
            end_value=end_value,
            x_axis=x_axis,
            font=font,
            title=title
        )
        return figure
    return None

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
        shared_xaxes=True,
        vertical_spacing=0.02
    )
    font = dict(size=20, color='#7f7f7f')
    x_axis_tickvals = sources[::2] if len(sources) > 2 else sources

    for i, source_map in enumerate(probability_map_list):
        support = np.array(source_map.columns)
        counts_subarray = [row for row in source_map.values]
        counts_subarray = list(zip(*counts_subarray))

        figure = go.Heatmap(
            x=sources,
            y=support,
            z=counts_subarray,
            reversescale=True,
            coloraxis='coloraxis'
        )

        subplot.add_trace(
            figure,
            row=i + 1,
            col=1
        )

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
            tickvals=x_axis_tickvals,
            tickfont={'size': 12},
            title_text='Source' if i == dimensions - 1 else None,
            title_font=font if i == dimensions - 1 else None,
            row=i + 1,
            col=1,
            ticks='outside',
            tickcolor='black'
        )

        subplot.update_layout(
            autosize=True,
            height=min(300 * dimensions, 800),
            showlegend=False,
            template='plotly_white',
            margin=dict(t=60, r=20, b=60, l=60),
            coloraxis=dict(colorscale='Spectral_r'),
            title=f'{"Absolute frequencies" if absolute else "Probability distribution"} '
                  f'data source heatmap'
        )

    subplot.show()
    return subplot

def plot_conditional_data_source_map(
        data_source_map_dict: Dict[str, MultiVariateDataSourceMap],
        absolute: bool = False
) -> None:
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
    None
    """

    if not isinstance(data_source_map_dict, dict) and not all(
            isinstance(value, MultiVariateDataSourceMap) for value in data_source_map_dict.values()
    ):
        raise TypeError('data_source_map_dict must be a dictionary with MultiVariateDataSourceMap instances, resultant'
                        'of the estimate_conditional_data_source_map function.')

    if not isinstance(absolute, bool):
        raise TypeError('absolute must be a boolean value, indicating whether to plot absolute counts or probabilities.')


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

    for dim in range(dimensions):
        subplot = sp.make_subplots(
            rows=len(labels),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04
        )

        font = dict(size=20, color='#7f7f7f')

        for label, probability_map_list in probability_map_dict.items():
            sources = sources_dict[label]
            x_axis_tickvals = sources[::2] if len(sources) > 2 else sources
            source_map = probability_map_list[dim]
            support = np.array(source_map.columns)
            counts_subarray = [row for row in source_map.values]
            counts_subarray = list(zip(*counts_subarray))

            figure = go.Heatmap(
                x=sources,
                y=support,
                z=counts_subarray,
                reversescale=True,
                coloraxis='coloraxis'
            )

            subplot.add_trace(figure, row=labels.index(label) + 1, col=1)

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
                tickvals=x_axis_tickvals,
                tickfont={'size': 12},
                title_text='Source' if labels.index(label) == len(labels) - 1 else None,
                title_font=font if labels.index(label) == len(labels) - 1 else None,
                row=labels.index(label) + 1,
                col=1,
                ticks='outside',
                tickcolor='black'
            )

        subplot.update_layout(
            autosize=True,
            height=min(300 * len(labels), 800),
            showlegend=False,
            template='plotly_white',
            margin=dict(t=60, r=20, b=60, l=60),
            coloraxis=dict(colorscale='Spectral_r'),
            title=f'{"Absolute frequencies" if absolute else "Probability distribution"} '
                  f'conditional data source heatmap of Principal Component {dim + 1}'
        )

        subplot.show()
    return None






