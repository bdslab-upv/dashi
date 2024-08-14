from datetime import datetime

import numpy as np
import plotly.graph_objs as go
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex

from datashift.constants import *
from datashift.igt.igt_projection import IGTProjection
from datashift.igt.igt_trajectory_estimator import estimate_igt_trajectory
from datashift.utils import matplotlib_to_plotly, format_date_for_year, format_date_for_month, format_date_for_week


def plot_IGT_projection(
        igt_projection: IGTProjection,
        dimensions: int = 3,
        start_date: datetime = None,
        end_date: datetime = None,
        color_palette: PlotColorPalette = PlotColorPalette.Spectral,
        trajectory: bool = False
):
    # Validate dimensions
    if dimensions not in [2, 3]:
        raise ValueError(
            'Currently IGT plot can only be made on 2 or 3 dimensions, please set dimensions parameter accordingly')

    if not isinstance(color_palette, PlotColorPalette) or \
            color_palette.name not in [palette.name for palette in PlotColorPalette]:
        raise ValueError('color_palette must be one of the defined in PlotColorPalette')

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
        color_map = get_cmap(color_palette.value)
        colors = [to_hex(color_map(i / len(dates))) for i in range(len(dates) + 1)]
    elif period in [TEMPORAL_PERIOD_MONTH, TEMPORAL_PERIOD_WEEK]:
        color_map = get_cmap(color_palette.value, 128)
        color_list = matplotlib_to_plotly(color_map)
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
                        hoverinfo='text',
                        marker=dict(
                            color=colors[i]
                        ),
                        text=format_date_for_year(dates[i]),
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
                        hoverinfo='text',
                        marker=dict(
                            color=period_colors[dates[i].month - 1]
                        ),
                        hovertext=f"{dates[i].strftime('%Y')}-{MONTH_LONG_ABBREVIATIONS[dates[i].month - 1]}",
                        text=format_date_for_month(dates[i]),
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
                        hoverinfo='text',
                        marker=dict(
                            color=period_colors[dates[i].isoweekday() - 1]
                        ),
                        text=format_date_for_week(dates[i]),
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
                        hoverinfo='text',
                        marker=dict(
                            color=colors[i]
                        ),
                        text=format_date_for_year(dates[i]),
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
                        hoverinfo='text',
                        hovertext=f"{dates[i].strftime('%Y')}-{MONTH_LONG_ABBREVIATIONS[dates[i].month - 1]}",
                        marker=dict(
                            color=period_colors[dates[i].month - 1]
                        ),
                        text=format_date_for_month(dates[i]),
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
                        hoverinfo='text',
                        marker=dict(
                            color=period_colors[dates[i].isoweekday() - 1]
                        ),
                        text=format_date_for_week(dates[i]),
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
                        color="#21908C", width=1.3,  # Color based on row index
                        showscale=False
                    ),
                    hovertext=[f"Approx. date: {date}" for date in trajectory_dates]
                )
            )

    fig.update_layout(
        plot_bgcolor='white',
        showlegend=False,
        title={
            'text': 'Information Geometric Temporal (IGT) projection',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
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
