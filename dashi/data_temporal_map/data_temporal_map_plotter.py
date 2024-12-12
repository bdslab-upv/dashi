"""
Data Temporal Map plotting main functions and classes
"""
# Author: David Fernández Narro <dfernar@upv.edu.es>
#         Ángel Sánchez García <ansan12a@upv.es>
#         Pablo Ferri Borredá <pabferb2@upv.es>
#         Carlos Sáez Silvestre <carsaesi@upv.es>
#         Juan Miguel García Gómez <juanmig@upv.es>

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
from typing import Optional, Dict

from dashi.constants import DataTemporalMapPlotSortingMethod, PlotColorPalette, \
    DataTemporalMapPlotMode, VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE
from dashi.data_temporal_map.data_temporal_map import DataTemporalMap, MultiVariateDataTemporalMap, \
    trim_data_temporal_map


def plot_data_temporal_map(
        data_temporal_map: DataTemporalMap,
        absolute: bool = False,
        log_transform: bool = False,
        start_value: Optional[int] = 0,
        end_value: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sorting_method: DataTemporalMapPlotSortingMethod = DataTemporalMapPlotSortingMethod.Frequency,
        color_palette: PlotColorPalette = PlotColorPalette.Spectral,
        mode: DataTemporalMapPlotMode = DataTemporalMapPlotMode.Heatmap,
        plot_title: Optional[str] = None
) -> go.Figure:
    """
    Plots a Data Temporal heatmap or series from a DataTemporalMap object.

    Parameters
    ----------
    data_temporal_map : DataTemporalMap
        The DataTemporalMap object that contains the temporal data to be plotted.

    absolute : bool
        If True, plot absolute values; otherwise, the relative probabilities are plotted. Default is False.

    log_transform : bool
        If True, applies a log transformation to the data for better visibility of small values. Default is False.

    start_value : int, optional
        The value at which to start the plot. Default is 0.

    end_value : int, optional
        The value at which to end the plot. If None, the plot extends to the last value. Default is None.

    start_date : datetime, optional
        The starting date for the plot (filters the data). If None, uses the first date in the data. Default is None.

    end_date : datetime, optional
        The ending date for the plot (filters the data). If None, uses the last date in the data. Default is None.

    sorting_method : DataTemporalMapPlotSortingMethod, optional
        The method by which to sort the data for visualization. Default is Frequency.

    color_palette : PlotColorPalette, optional
        The color palette to be used for the plot. Default is Spectral.

    mode : DataTemporalMapPlotMode, optional
        The mode of visualization (e.g., Heatmap, Series). Default is Heatmap.

    plot_title : str, optional
        The title of the plot. If None, a default title is used. Default is None.

    Returns
    -------
    Figure
        The Plotly figure object representing the plot
    """
    if not type(data_temporal_map) == DataTemporalMap:
        raise TypeError('data_temporal_map must be of type DataTemporalMap. For multivariate plot'
                        ' use plot_multivariate_data_temporal_map function')

    if not isinstance(mode, DataTemporalMapPlotMode) or \
            mode.name not in [mode.name for mode in DataTemporalMapPlotMode]:
        raise ValueError(f'mode must be one of the defined in DataTemporalMapPlotMode')

    if not isinstance(color_palette, PlotColorPalette) or \
            color_palette.name not in [palette.name for palette in PlotColorPalette]:
        raise ValueError('color_palette must be one of the defined in PlotColorPalette')

    if not isinstance(absolute, bool):
        raise ValueError('absolute must be a logical value')

    if not isinstance(log_transform, bool):
        raise ValueError('log_transform must be a logical value')

    if not isinstance(start_value, int) and start_value < 1:
        raise ValueError('start_value must be greater or equal than 1')

    if not isinstance(sorting_method, DataTemporalMapPlotSortingMethod) or \
            sorting_method.name not in [method.name for method in DataTemporalMapPlotSortingMethod]:
        raise ValueError('sorting_method must be one of the defined in DataTemporalMapPlotSortingMethod')

    # TODO: check color scales for heatmap and series
    color_scale = color_palette.value

    data_temporal_map = trim_data_temporal_map(data_temporal_map, start_date, end_date)

    if absolute:
        temporal_map = data_temporal_map.counts_map
    else:
        temporal_map = data_temporal_map.probability_map

    dates = data_temporal_map.dates

    support = np.array(data_temporal_map.support.iloc[:, 0].tolist())
    variable_type = data_temporal_map.variable_type

    if variable_type in [VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE]:
        if sorting_method == DataTemporalMapPlotSortingMethod.Frequency:
            support_order = np.argsort(np.sum(temporal_map, axis=0))[::-1]
        else:
            support_order = np.argsort(support)

        support = support[support_order]

        # Resort temporal map by support_order
        temporal_map = [row[support_order] for row in temporal_map]
        temporal_map = np.array(temporal_map)

        any_supp_na = pd.isnull(support)
        if any_supp_na.any():
            support[any_supp_na] = '<NA>'

    if not end_value or end_value > temporal_map.shape[0]:
        end_value = temporal_map.shape[1]

    if start_value > temporal_map.shape[0]:
        start_value = temporal_map.shape[1]

    font = dict(size=20, color='#7f7f7f')
    x_axis = dict(title='Date',
                  tickvals=dates,
                  titlefont=font,
                  type='date')

    if log_transform:
        temporal_map = np.log(temporal_map)

    counts_subarray = [row[start_value:end_value] for row in temporal_map]
    counts_subarray = list(zip(*counts_subarray))  # Transpose the matrix

    if mode == DataTemporalMapPlotMode.Heatmap:
        figure = go.Figure(
            data=go.Heatmap(
                x=dates,
                y=support[start_value:end_value],
                z=counts_subarray,
                colorscale=color_scale,
                reversescale=True
            )
        )

        y_axis = dict(
            title=data_temporal_map.variable_name,
            titlefont=font,
            automargin=True,
        )

        figure.update_layout(xaxis=x_axis, yaxis=y_axis)

        # Avoid type casting in plotly
        if variable_type in [VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE]:
            figure.update_layout(yaxis_type='category')

        if plot_title is not None:
            figure.update_layout(title=plot_title)
        else:
            plot_title = 'Probability distribution data temporal heatmap'
            if absolute:
                plot_title = 'Absolute frequencies data temporal heatmap'
            figure.update_layout(title=plot_title)

    elif mode == DataTemporalMapPlotMode.Series:
        figure = go.Figure()

        for i in range(start_value, end_value):
            trace = go.Scatter(
                x=dates,
                y=counts_subarray[i],
                mode='lines',
                name=str(support[i])
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
        )

        figure.update_layout(
            xaxis=x_axis,
            yaxis=y_axis,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        if plot_title is not None:
            figure.update_layout(title=plot_title)
        else:
            figure.update_layout(title='Evolution of ' + data_temporal_map.variable_name)

    figure.show()
    return figure


def plot_multivariate_data_temporal_map(
        data_temporal_map: MultiVariateDataTemporalMap,
        absolute: bool = False,
        plot_title: str = None
) -> go.Figure:
    """
    Plots a Data Temporal heatmap from a MultiVariateDataTemporalMap object.

    Parameters
    ----------
    data_temporal_map : MultiVariateDataTemporalMap
        The MultiVariateDataTemporalMap object that contains the temporal data to be plotted.

    absolute : bool, optional
        If True, plot absolute values; otherwise, the relative probabilities are plotted. Default is False.

    plot_title : str, optional
        The title of the plot. If None, a default title is used. Default is None.

    Returns
    -------
    Figure
        The Plotly figure object representing the plot.
    """
    if not type(data_temporal_map) == MultiVariateDataTemporalMap:
        raise TypeError('data_temporal_map must be of type MultiVariateDataTemporalMap. For univariate plot use'
                        ' plot_data_temporal_map function')

    if not isinstance(absolute, bool):
        raise ValueError('absolute must be a logical value')

    dates = data_temporal_map.dates

    supports = data_temporal_map.multivariate_support
    dimensions = len(supports)

    if absolute:
        multivariate_map = data_temporal_map.multivariate_counts_map
    else:
        multivariate_map = data_temporal_map.multivariate_probability_map

    probability_map_list = list()

    if dimensions == 2:
        probability_map_dim1 = pd.DataFrame([np.sum(dim1, axis=0) for dim1 in multivariate_map],
                                            columns=supports[0])
        probability_map_dim2 = pd.DataFrame([np.sum(dim2, axis=1) for dim2 in multivariate_map],
                                            columns=supports[1])
        probability_map_list.extend([probability_map_dim1, probability_map_dim2])

    elif dimensions == 3:
        probability_map_dim1 = pd.DataFrame([np.sum(dim1, axis=(2, 0)) for dim1 in multivariate_map],
                                            columns=supports[0])
        probability_map_dim2 = pd.DataFrame([np.sum(dim2, axis=(2, 1)) for dim2 in multivariate_map],
                                            columns=supports[1])
        probability_map_dim3 = pd.DataFrame([np.sum(dim3, axis=(0, 1)) for dim3 in multivariate_map],
                                            columns=supports[2])
        probability_map_list.extend([probability_map_dim1, probability_map_dim2, probability_map_dim3])

    subplot = sp.make_subplots(rows=dimensions,
                               cols=1,
                               shared_xaxes=True,
                               vertical_spacing=0.02
                               )

    font = dict(size=20, color='#7f7f7f')
    x_axis = dict(
        tickvals=dates,
        tickfont=font,
        type='date')

    for i, temporal_map in enumerate(probability_map_list):
        support = np.array(temporal_map.columns)
        counts_subarray = [row for row in temporal_map.values]
        counts_subarray = list(zip(*counts_subarray))

        figure = go.Heatmap(
            x=dates,
            y=support,
            z=counts_subarray,
            reversescale=True,
            coloraxis='coloraxis'
        )

        subplot.add_trace(figure, row=i + 1, col=1)

        y_axis = dict(
            title=f'PC {i + 1}',
            titlefont=font,
            automargin=True,
        )

        subplot.update_yaxes(y_axis, row=i + 1, col=1)

        if i == dimensions - 1:
            subplot.update_xaxes(title_text="Date", title_font=font, row=i + 1, col=1)
        else:
            subplot.update_xaxes(x_axis, row=i + 1, col=1)

    subplot.update_layout(xaxis=x_axis,
                          autosize=True,
                          height=min(300 * dimensions, 800),
                          showlegend=False,
                          template='plotly_white',
                          margin=dict(t=60, r=20, b=60, l=60),
                          coloraxis=dict(
                              colorscale='Spectral_r'
                          )
                          )
    if plot_title is not None:
        subplot.update_layout(title=plot_title)
    else:
        plot_title = 'Probability distribution data temporal heatmap'
        if absolute:
            plot_title = 'Absolute frequencies data temporal heatmap'
        subplot.update_layout(title=plot_title)

    subplot.show()
    return subplot


def plot_multivariate_concept_shift(
        data_temporal_map_dict: Dict[str, MultiVariateDataTemporalMap],
        absolute: bool = False,
):
    """
    Plots a Figure for each dimension selected in the data_temporal_map_dict. Each Figure represents the
    Data Temporal heatmap of each label in that dimension

    Parameters
    ----------
    data_temporal_map_dict : Dict[str, MultiVariateDataTemporalMap]
        A dictionary where keys are labels (strings), and values are the corresponding
        `MultiVariateDataTemporalMap` objects obtained from the 'estimate_multidim_concept_shift' function.

    absolute : bool, optional
        If True, plot absolute values; otherwise, relative probabilities are plotted. Default is False.

    Returns
    -------
    None
    """
    if not type(data_temporal_map_dict) == dict:
        raise TypeError('data_temporal_map must be a dictionary of objects MultiVariateDataTemporalMap, resultant of '
                        'the estimate_multidim_concept_shift function')

    if not isinstance(absolute, bool):
        raise ValueError('absolute must be a logical value')

    labels = list(data_temporal_map_dict.keys())
    probability_map_dict = dict()
    for label, data_temporal_map in data_temporal_map_dict.items():
        dates = data_temporal_map.dates

        supports = data_temporal_map.multivariate_support
        dimensions = len(supports)

        if absolute:
            multivariate_map = data_temporal_map.multivariate_counts_map
        else:
            multivariate_map = data_temporal_map.multivariate_probability_map

        probability_map_list = list()

        if dimensions == 2:
            probability_map_dim1 = pd.DataFrame([np.sum(dim1, axis=0) for dim1 in multivariate_map],
                                                columns=supports[0])
            probability_map_dim2 = pd.DataFrame([np.sum(dim2, axis=1) for dim2 in multivariate_map],
                                                columns=supports[1])
            probability_map_list.extend([probability_map_dim1, probability_map_dim2])

        elif dimensions == 3:
            probability_map_dim1 = pd.DataFrame([np.sum(dim1, axis=(2, 0)) for dim1 in multivariate_map],
                                                columns=supports[0])
            probability_map_dim2 = pd.DataFrame([np.sum(dim2, axis=(2, 1)) for dim2 in multivariate_map],
                                                columns=supports[1])
            probability_map_dim3 = pd.DataFrame([np.sum(dim3, axis=(0, 1)) for dim3 in multivariate_map],
                                                columns=supports[2])
            probability_map_list.extend([probability_map_dim1, probability_map_dim2, probability_map_dim3])

        probability_map_dict[label] = probability_map_list

    for dim in range(dimensions):
        subplot = sp.make_subplots(rows=len(labels),
                                   cols=1,
                                   shared_xaxes=True,
                                   vertical_spacing=0.02
                                   )

        font = dict(size=20, color='#7f7f7f')
        x_axis = dict(
            tickvals=dates,
            tickfont=font,
            type='date'
        )

        for label, probability_map_list in probability_map_dict.items():
            temporal_map = probability_map_list[dim]
            support = np.array(temporal_map.columns)
            counts_subarray = [row for row in temporal_map.values]
            counts_subarray = list(zip(*counts_subarray))

            figure = go.Heatmap(
                x=dates,
                y=support,
                z=counts_subarray,
                reversescale=True,
                coloraxis='coloraxis'
            )

            subplot.add_trace(figure, row=labels.index(label) + 1, col=1)

            y_axis = dict(
                title=f'Label: {label}',
                titlefont=font,
                automargin=True,
            )

            subplot.update_yaxes(y_axis, row=labels.index(label) + 1, col=1)

            if labels.index(label) == len(labels) - 1:
                subplot.update_xaxes(title_text='Date', title_font=font, row=labels.index(label) + 1, col=1)
            else:
                subplot.update_xaxes(x_axis, row=labels.index(label) + 1, col=1)

        subplot.update_layout(xaxis=x_axis,
                              autosize=True,
                              height=min(300 * len(labels), 800),
                              showlegend=False,
                              template='plotly_white',
                              margin=dict(t=60, r=20, b=60, l=60),
                              coloraxis=dict(
                                    colorscale='Spectral_r'
                              )
                              )

        plot_title = f'Probability distribution data temporal heatmap of Principal Component {dim + 1}'
        if absolute:
            plot_title = f'Absolute frequencies data temporal heatmap of Principal Component {dim + 1}'
        subplot.update_layout(title=plot_title)

        subplot.show()






