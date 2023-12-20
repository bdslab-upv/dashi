import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

from all_constants import VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE


def plot_data_temporal_map(
        data_temporal_map,
        absolute,
        start_value,
        end_value,
        start_date,
        end_date,
        sorting_method,
        color_palette,
        mode,
        plot_title=None
):
    if mode not in ['heatmap', 'series']:
        raise ValueError('mode must be one of heatmap or series')

    if color_palette not in ['Spectral', 'Viridis', 'Magma', 'Viridis-reversed', 'Magma-reversed']:
        raise ValueError('color_palette must be one of Spectral, Viridis, Magma, Viridis-reversed or Magma-reversed')

    if not isinstance(absolute, bool):
        raise ValueError('absolute must be a logical value')

    # TODO: check this
    # if start_value < 1:
    #     raise ValueError('start_value must be greater or equal than 1')

    if sorting_method not in ['frequency', 'alphabetical']:
        raise ValueError('sorting_method must be one of frequency or alphabetical')

    # TODO: check color scales for heatmap and series
    color_scale = {
        'Spectral': 'Spectral',
        'Viridis': 'Viridis',
        'Magma': 'Magma',
        'Viridis-reversed': 'Viridis_r',
        'Magma-reversed': 'Magma_r'
    }[color_palette]

    if absolute:
        temporal_map = data_temporal_map.counts_map
    else:
        temporal_map = data_temporal_map.probability_map

    temporal_map_type = data_temporal_map.variable_type

    dates = data_temporal_map.dates

    support = np.array(data_temporal_map.support.iloc[:, 0].tolist())
    variable_type = data_temporal_map.variable_type

    if variable_type in [VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE]:
        if sorting_method == 'frequency':
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

    if end_value > temporal_map.shape[0]:
        end_value = temporal_map.shape[0]

    font = dict(size=18, color='#7f7f7f')
    x_axis = dict(title='Date', titlefont=font, type='date')

    counts_subarray = [row[start_value:end_value] for row in temporal_map]
    counts_subarray = list(zip(*counts_subarray))  # Transpose the matrix

    if mode == 'heatmap':
        figure = go.figureure(
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
        if temporal_map_type in [VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE]:
            figure.update_layout(yaxis_type='category')

        if plot_title is not None:
            figure.update_layout(title=plot_title)
        else:
            plot_title = 'Probability distribution data temporal heatmap'
            if absolute:
                'Absolute frequencies data temporal heatmap'
            figure.update_layout(title=plot_title)

    else:  # mode == 'series'
        figure = go.figureure()
        max_colors = 6

        for i in range(start_value, end_value):
            trace = go.Scatter(
                x=dates,
                y=counts_subarray[i],
                mode='lines',
                name=support[i]
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
        )

        if plot_title is not None:
            figure.update_layout(title=plot_title)
        else:
            figure.update_layout(title='Evolution of ' + data_temporal_map.variable_name)

    figure.show()
    return figure
