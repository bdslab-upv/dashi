from datetime import datetime

import matplotlib
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex

from all_classes import IGTProjection
from all_constants import VALID_STRING_TYPE, VALID_CATEGORICAL_TYPE, TEMPORAL_PERIOD_YEAR, TEMPORAL_PERIOD_WEEK, \
    TEMPORAL_PERIOD_MONTH, MONTH_SHORT_ABBREVIATIONS, MONTH_LONG_ABBREVIATIONS
from estimate_igt_trajectory import estimate_igt_trajectory


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
        figure = go.Figure()
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


def plot_IGT_projection(
        igt_projection: IGTProjection,
        dimensions: int = 3,
        start_date: datetime = None,
        end_date: datetime = None,
        color_palette: str = "Spectral",
        trajectory: bool = False
):
    # Validate dimensions
    if dimensions not in [2, 3]:
        raise ValueError(
            'Currently IGT plot can only be made on 2 or 3 dimensions, please set dimensions parameter accordingly')

    # TODO To check
    # Validate color palette
    valid_palettes = ["Spectral", "Viridis", "Magma", "Viridis-reversed", "Magma-reversed"]
    if color_palette not in valid_palettes:
        raise ValueError("color_palette must be one of " + ", ".join(valid_palettes))

    if not start_date:
        start_date = min(igt_projection.data_temporal_map.dates)
    if not end_date:
        end_date = max(igt_projection.data_temporal_map.dates)

    # Date filtering
    date_mask = (igt_projection.data_temporal_map.dates >= np.datetime64(start_date)) & (
            igt_projection.data_temporal_map.dates <= np.datetime64(end_date))
    dates = igt_projection.data_temporal_map.dates[date_mask]
    projection = igt_projection.projection[date_mask]

    # Estimating trajectory if needed
    if trajectory:
        igt_trajectory = estimate_igt_trajectory(igt_projection)
        trajectory_points = igt_trajectory['points']
        trajectory_dates = igt_trajectory['dates']

    # Generate colors for ten data points

    # TODO: colors
    # Set color based on period
    period = igt_projection.data_temporal_map.period
    colors = []
    period_colors = []

    if period == TEMPORAL_PERIOD_YEAR:
        color_map = get_cmap(color_palette)
        colors = [to_hex(color_map(i / len(dates))) for i in range(len(dates) + 1)]
    elif period in [TEMPORAL_PERIOD_MONTH, TEMPORAL_PERIOD_WEEK]:
        color_map = get_cmap(color_palette, 128)
        color_list = __matplotlib_to_plotly(color_map)
        color_list.reverse()

        days_of_period = 12 if period == TEMPORAL_PERIOD_MONTH else 53

        color_list.extend(reversed(color_list))
        colors = np.array(color_list)

        period_indexes = np.round(np.linspace(0, 255, days_of_period)).astype(int)
        period_colors = colors.take([period_indexes])[0]

    fig = go.Figure()

    # Plotting
    if dimensions == 2:
        if period == TEMPORAL_PERIOD_YEAR:
            # Add scatter for each point
            for i, (x, y) in enumerate(projection):
                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        mode='text',
                        marker=dict(
                            color=colors[i]
                        ),
                        text=__format_date_for_year(dates[i]),
                        textposition="top center",
                        textfont_color=colors[i]
                    )
                )
        elif period == TEMPORAL_PERIOD_MONTH:
            # Add scatter for each point
            for i, (x, y) in enumerate(projection):
                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        mode='text',
                        marker=dict(
                            color=period_colors[dates[i].month - 1]
                        ),
                        hovertext=f"{dates[i].strftime('%Y')}-{MONTH_LONG_ABBREVIATIONS[dates[i].month - 1]}",
                        text=__format_date_for_month(dates[i]),
                        textposition="top center",
                        textfont_color=period_colors[dates[i].month - 1]
                    )
                )
        elif period == TEMPORAL_PERIOD_WEEK:
            # Add scatter for each point
            for i, (x, y) in enumerate(projection):
                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        mode='text',
                        marker=dict(
                            color=period_colors[dates[i].isoweekday() - 1]
                        ),
                        text=__format_date_for_week(dates[i]),
                        textposition="top center",
                        textfont_color=period_colors[dates[i].isoweekday() - 1]
                    )
                )

        # Add trajectory if necessary
        if trajectory:
            fig.add_trace(
                go.Scatter(
                    x=trajectory_points['D1'],
                    y=trajectory_points['D2'],
                    mode='lines',
                    line=dict(color="#21908C", width=1),
                    hovertext=[f"Approx. date: {date}" for date in trajectory_dates]
                )
            )
    elif dimensions == 3:
        if period == TEMPORAL_PERIOD_YEAR:
            # Add scatter for each point
            for i, (x, y, z) in enumerate(projection):
                fig.add_trace(
                    go.Scatter3d(
                        x=[x],
                        y=[y],
                        z=[z],
                        mode='text',
                        marker=dict(
                            color=colors[i]
                        ),
                        text=__format_date_for_year(dates[i]),
                        textposition="top center",
                        textfont_color=colors[i]
                    )
                )
        elif period == TEMPORAL_PERIOD_MONTH:
            # Add scatter for each point
            for i, (x, y, z) in enumerate(projection):
                fig.add_trace(
                    go.Scatter3d(
                        x=[x],
                        y=[y],
                        z=[z],
                        mode='text',
                        marker=dict(
                            color=period_colors[dates[i].month - 1]
                        ),
                        text=__format_date_for_month(dates[i]),
                        textposition="top center",
                        textfont_color=period_colors[dates[i].month - 1]
                    )
                )
        elif period == TEMPORAL_PERIOD_WEEK:
            # Add scatter for each point
            for i, (x, y, z) in enumerate(projection):
                fig.add_trace(
                    go.Scatter3d(
                        x=[x],
                        y=[y],
                        z=[z],
                        mode='text',
                        marker=dict(
                            color=period_colors[dates[i].isoweekday() - 1]
                        ),
                        text=__format_date_for_week(dates[i]),
                        textposition="top center",
                        textfont_color=period_colors[dates[i].isoweekday() - 1]
                    )
                )
        # Add trajectory if necessary
        if trajectory:
            fig.add_trace(
                go.Scatter3d(
                    x=trajectory_points['D1'],
                    y=trajectory_points['D2'],
                    z=trajectory_points['D3'],
                    mode='lines',
                    line=dict(
                        color=np.arange(0, len(trajectory_points)),  # Color based on row index
                        showscale=False
                    ),
                    hovertext=[f"Approx. date: {date}" for date in trajectory_dates]
                )
            )

    fig.update_layout(
        plot_bgcolor='white',
        scene=dict(
            xaxis=dict(
                title='D1',
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="lightgrey",
                showbackground=True,
                zerolinecolor="black"
            ),
            yaxis=dict(
                title='D2',
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="lightgrey",
                showbackground=True,
                zerolinecolor="black"
            ),
            zaxis=dict(
                title='D3',
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="lightgrey",
                showbackground=True,
                zerolinecolor="black"
            ),
        ),
    )
    fig.update_xaxes(
        title='D1',
        mirror=True,
        ticks='outside',
        showline=True,
        gridcolor='lightgrey',
        zerolinecolor='black'
    )
    fig.update_yaxes(
        title='D2',
        mirror=True,
        ticks='outside',
        showline=True,
        gridcolor='lightgrey',
        zerolinecolor='black'
    )
    fig.show()


def __format_date_for_year(date: datetime) -> str:
    year_part = date.strftime('%y')

    return year_part


def __format_date_for_month(date: datetime) -> str:
    year_part = date.strftime('%y')
    month_part = MONTH_SHORT_ABBREVIATIONS[date.month - 1]

    return year_part + month_part


def __format_date_for_week(date: datetime) -> str:
    year_part = date.strftime('%y')
    month_part = MONTH_SHORT_ABBREVIATIONS[date.month - 1]
    day_part = str(date.isoweekday())

    return year_part + month_part + day_part


def __matplotlib_to_plotly(cmap):
    pl_colorscale = []

    for k in range(128):
        C = np.array([cmap(k)[0] * 255, cmap(k)[1] * 255, cmap(k)[2] * 255])
        pl_colorscale.append(f'rgb({C[0]}, {C[1]}, {C[2]})')

    return pl_colorscale
