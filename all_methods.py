import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px


#TODO: write plot title
def plot_data_temporal_map(data_temporal_map, absolute, start_value, end_value, start_date, end_date, sorting_method,
                           color_palette, mode):
    if mode not in ['heatmap', 'series']:
        raise ValueError("mode must be one of heatmap or series")

    if color_palette not in ['Spectral', 'Viridis', 'Magma', 'Viridis-reversed', 'Magma-reversed']:
        raise ValueError("color_palette must be one of Spectral, Viridis, Magma, Viridis-reversed or Magma-reversed")

    if not isinstance(absolute, bool):
        raise ValueError("absolute must be a logical value")

    # if start_value < 1:
    #     raise ValueError("start_value must be greater or equal than 1")

    if sorting_method not in ['frequency', 'alphabetical']:
        raise ValueError("sorting_method must be one of frequency or alphabetical")

    color_scale = {
        'Spectral': 'Spectral',
        'Viridis': 'Viridis',
        'Magma': 'Magma',
        'Viridis-reversed': 'Viridis_r',
        'Magma-reversed': 'Magma_r'
    }[color_palette]

    temporal_map = data_temporal_map.probability_map if not absolute else data_temporal_map.counts_map

    dates = data_temporal_map.dates
    start_idx = np.where(dates == start_date)[0][0]
    end_idx = np.where(dates == end_date)[0][0] + 1

    support = np.array(data_temporal_map.support.iloc[:, 0].tolist())
    variable_type = data_temporal_map.variable_type

    if variable_type in ['factor', 'object']:
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
            support[any_supp_na] = "<NA>"

    if end_value > temporal_map.shape[0]:
        end_value = temporal_map.shape[0]

    font = dict(size=18, color="#7f7f7f")
    x_axis = dict(title="Date", titlefont=font, type="date")
    y_axis = dict(
        title=data_temporal_map.variable_name,
        titlefont=font,
    #    tickfont=14,
        automargin=True,
        #type="category" if variable_type in ['character', 'factor'] else "-"
    )

    if mode == 'heatmap':
        counts_subarray = [row[start_value:end_value] for row in temporal_map]
        counts_subarray = list(zip(*counts_subarray))  # Transpose the matrix

        fig = go.Figure(
            data=go.Heatmap(
                x=dates,
                y=support[start_value:end_value],
                z=counts_subarray,
                colorscale=color_scale,
                reversescale=True
            )
        )
        fig.update_layout(xaxis=x_axis, yaxis=y_axis,
                        title="Absolute frequencies data temporal heatmap" if absolute else "Probability distribution data temporal heatmap",
                        #margin=margin
                          )
    else:  # mode == 'series'
        fig = go.Figure()
        max_colors = 6
        color_scale_values = {
            'Viridis': 'Viridis',
            'Magma': 'Magma',
            'Viridis-reversed': 'Viridis_r',
            'Magma-reversed': 'Magma_r'
        }[color_palette]

        for i in range(start_value, end_value + 1):
            trace = go.Scatter(x=dates[start_idx:end_idx], y=temporal_map[i - start_value, start_idx:end_idx],
                               name=support[i][0], mode='lines',
                               line_color=color_scale_values[(i - start_value) % max_colors])
            fig.add_trace(trace)

        fig.update_layout(
            title='Evolution of ' + data_temporal_map.variable_name,
            xaxis=x_axis,
            yaxis=y_axis,
            #margin=margin
        )
    fig.show()