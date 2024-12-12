"""
Information Geometric Temporal (IGT) plotting main functions and classes
"""
# Author: David Fernández Narro <dfernar@upv.edu.es>
#         Ángel Sánchez García <ansan12a@upv.es>
#         Pablo Ferri Borredá <pabferb2@upv.es>
#         Carlos Sáez Silvestre <carsaesi@upv.es>
#         Juan Miguel García Gómez <juanmig@upv.es>

from datetime import datetime
from typing import Optional

import numpy as np
import plotly.graph_objs as go
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex

from dashi.constants import *
from dashi.igt.igt_projection import IGTProjection
from dashi.igt.igt_trajectory_estimator import _estimate_igt_trajectory
from dashi.utils import matplotlib_to_plotly, format_date_for_year, format_date_for_month, format_date_for_week


def plot_IGT_projection(
        igt_projection: IGTProjection,
        dimensions: int = 2,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        color_palette: PlotColorPalette = PlotColorPalette.Spectral,
        trajectory: bool = False
) -> go.Figure:
    """
    Plots an interactive Information Geometric Temporal (IGT) plot from an \code{IGTProjection} object.
    An IGT plot visualizes the variability among time batches in a data repository in a 2D or 3D plot.
    Time batches are positioned as points where the distance between them represents the probabilistic
    distance between their distributions (currently Jensen-Shannon distance).
    To track the temporal evolution, temporal batches are labeled to show their date and
    colored according to their season or period, according to the analysis period, as follows.
    If period=="year" the label is "yy" (2 digit year) and the color is according to year.
    If period=="month" the label is "yym" (yy + abbreviated month*) and the color is according
    to the season (yearly).
    If period=="week" the label is "yymmw" (yym + ISO week number in 1-2 digit) and the color is
    according to the season (yearly). An estimated smoothed trajectory of the information evolution
    over time can be shown using the optional "trajectory" parameter.

    Note that since the projection is based on multidimensional scaling, a 2 dimensional
    projection entails a loss of information compared to a 3 dimensional projection. E.g., periodic
    variability components such as seasonal effect can be hindered by an abrupt change or a general trend.

    Parameters
    ----------
    igt_projection : IGTProjection
        The `IGTProjection` object containing the data for the temporal plot.

    dimensions : int, optional
        The number of dimensions to be used for plotting (2D or 3D). Default is 2.

    start_date : Optional[datetime], optional
        The starting date for the temporal plot. If None, it is not constrained. Default is None.

    end_date : Optional[datetime], optional
        The ending date for the temporal plot. If None, it is not constrained. Default is None.

    color_palette : PlotColorPalette, optional
        The color palette to be used for coloring the points. Default is Spectral.

    trajectory : bool, optional
        If True, a smoothed trajectory showing the evolution of the information across time is plotted.
        Default is False.

    Returns
    -------
    Figure
        The Plotly figure object containing the IGT projection plot.
    """
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
        igt_trajectory = _estimate_igt_trajectory(igt_projection)
        trajectory_points = igt_trajectory['points']
        trajectory_dates = igt_trajectory['dates']

    # Generate colors for ten data points
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

        title = {
            'text': 'Information Geometric Temporal (IGT) projection',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
        }

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

        title = {
            'text': 'Information Geometric Temporal (IGT) projection',
            'x': 0.5,
            'y': 0.9,
            'xanchor': 'center',
            'yanchor': 'top',
        }

    fig.update_layout(
        plot_bgcolor='white',
        showlegend=False,
        title=title,
        margin=dict(
            l=40,
            r=40,
            b=50,
            t=90
        ),
        scene=dict(
            xaxis=dict(
                title='D1',
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="lightgrey",
                showbackground=True,
                zerolinecolor="black"
            ),
            yaxis=dict(
                title='D2',
                backgroundcolor="rgba(0, 0, 0, 0)",
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
    return fig
