from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from datashift.constants import DataTemporalMapPlotSortingMethod, PlotColorPalette, \
    DataTemporalMapPlotMode, VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE
from datashift.data_temporal_map.data_temporal_map import DataTemporalMap, trim_data_temporal_map


def plot_data_temporal_map(
        data_temporal_map: DataTemporalMap,
        absolute: bool = False,
        log_transform: bool = False,
        start_value: int = 0,
        end_value: int = None,
        start_date: datetime = None,
        end_date: datetime = None,
        sorting_method: DataTemporalMapPlotSortingMethod = DataTemporalMapPlotSortingMethod.Frequency,
        color_palette: PlotColorPalette = PlotColorPalette.Spectral,
        mode: DataTemporalMapPlotMode = DataTemporalMapPlotMode.Heatmap,
        plot_title: str = None
):
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

    font = dict(size=18, color='#7f7f7f')
    x_axis = dict(title='Date', titlefont=font, type='date')

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
                'Absolute frequencies data temporal heatmap'
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