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
Main module for plotting Multi Source Variability (MSV) metrics.
"""
import numbers
import numpy as np
import plotly.graph_objects as go

from dashi._constants import VALID_COLOR_PALETTES
from dashi.unsupervised_characterization.variability_metrics.estimate_msv_metrics import MSVMetrics

__all__ = ['plot_MSV']


def plot_MSV(
    msv_metrics: MSVMetrics,
    dimensions: int = 1,
    color_palette: str = 'Spectral',
    scale_factor: str | float = 'auto',
) -> go.Figure:
    """
    Plots a Multi Source Variability (MSV) metrics visualization from a ``MSVMetrics`` object.

    Parameters
    ----------
    msv_metrics : MSVMetrics
        An instance of the `MSVMetrics` class containing the metrics to be plotted.

    dimensions : int, optional
        The number of dimensions for the plot. Must be 1, 2, or 3. Default is 1.

    color_palette : str, optional
        The color palette to use for the plot (e.g., 'Spectral', 'viridis', 'viridis_r', 'magma', 'magma_r).
        Default is 'Spectral'.

    scale_factor : {'auto', float}, optional
        Marker size scaling factor.
        - 'auto' (default).
        - float > 0: user-provided scale factor used directly.

    Returns
    -------
    go.Figure
        A Plotly figure object containing the MSV metrics visualization.
    """

    # ---------- Input validation ----------
    if dimensions not in [1, 2, 3]:
        raise ValueError('Dimensions must be 1, 2, or 3.')

    if dimensions >= len(msv_metrics.sources):
        raise ValueError(
            'Dimensions must go from 1 to the number of sources - 1. Number of sources: '
            f'{len(msv_metrics.sources)}'
        )

    if color_palette not in VALID_COLOR_PALETTES:
        raise ValueError(f'Invalid color palette. Choose from: {VALID_COLOR_PALETTES}')

    if isinstance(scale_factor, str):
        if scale_factor != 'auto':
            raise ValueError("scale_factor must be 'auto' or a positive numeric value.")
    elif isinstance(scale_factor, numbers.Real):
        if not np.isfinite(scale_factor) or scale_factor <= 0:
            raise ValueError("Numeric scale_factor must be finite and > 0.")
    else:
        raise ValueError("scale_factor must be 'auto' or a positive numeric value.")

    # ---------- Extract data ----------
    vertices = np.asarray(msv_metrics.vertices)
    spos = np.asarray(msv_metrics.SPO)
    n_by_source = np.asarray(msv_metrics.nBySource, dtype=float)
    id_source = np.asarray(msv_metrics.sources)

    # ---------- Compute effective scale factor ----------
    sphere_max_size = 100.0

    finite_positive_mask = np.isfinite(n_by_source) & (n_by_source > 0)
    if scale_factor == 'auto':
        if np.any(finite_positive_mask):
            max_n = np.max(n_by_source[finite_positive_mask])
            effective_scale_factor = sphere_max_size / max_n
        else:
            effective_scale_factor = 0.0
    else:
        effective_scale_factor = float(scale_factor)

    # ---------- Compute marker sizes ----------
    sizes = np.where(
        np.isfinite(n_by_source) & (n_by_source > 0),
        n_by_source * effective_scale_factor,
        0.0
    )

    # ---------- Filter out zero-size points ----------
    plot_mask = sizes > 0

    if not np.all(plot_mask):
        removed_sources = id_source[~plot_mask]
        print(
            f"[plot_MSV] Excluding {removed_sources.size} source(s) with no data: "
            f"{removed_sources.tolist()}"
        )

    vertices_plot = vertices[plot_mask]
    spos_plot = spos[plot_mask]
    sizes_plot = sizes[plot_mask]
    id_source_plot = id_source[plot_mask]

    title = {
        'text': 'Multi Source Variability (MSV) Metrics',
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'color': 'black'}
    }

    # If nothing remains to plot, return an empty informative figure
    if vertices_plot.shape[0] == 0:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template='plotly_white',
            margin=dict(l=0, r=0, b=0, t=30),
            annotations=[
                dict(
                    text='No points to display (all marker sizes are 0).',
                    x=0.5, y=0.5, xref='paper', yref='paper',
                    showarrow=False, font=dict(color='black')
                )
            ],
        )
        return fig

    # ---------- Plot ----------
    if dimensions == 1:
        fig = go.Figure(
            data=go.Scatter(
                x=vertices_plot[:, 0],
                y=[0] * len(vertices_plot),  # y-coordinates are zero for 1D
                mode='markers+text',
                marker=dict(
                    size=sizes_plot,
                    sizemode='diameter',
                    color=spos_plot,
                    colorscale=color_palette,
                    opacity=0.8,
                    colorbar=dict(title='SPOs')
                ),
                text=id_source_plot,
                textposition='top center',
                hovertemplate='<b>%{text}</b><br>'
                              'x: %{x:.2f}<br>'
                              'SPO: %{marker.color:.2f}<extra></extra>'
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title='D1',
            yaxis_title='D2',
            margin=dict(l=0, r=0, b=0, t=30),
            template='plotly_white',
        )

    elif dimensions == 2:
        fig = go.Figure(
            data=go.Scatter(
                x=vertices_plot[:, 0],
                y=vertices_plot[:, 1],
                mode='markers+text',
                marker=dict(
                    size=sizes_plot,
                    sizemode='diameter',
                    color=spos_plot,
                    colorscale=color_palette,
                    opacity=0.8,
                    colorbar=dict(title='SPOs')
                ),
                text=id_source_plot,
                textposition='top center',
                hovertemplate='<b>%{text}</b><br>'
                              'x: %{x:.2f}<br>y: %{y:.2f}<br>'
                              'SPO: %{marker.color:.2f}<extra></extra>'
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title='D1',
            yaxis_title='D2',
            margin=dict(l=0, r=0, b=0, t=30),
            template='plotly_white',
        )

    else:
        fig = go.Figure(
            data=go.Scatter3d(
                x=vertices_plot[:, 0],
                y=vertices_plot[:, 1],
                z=vertices_plot[:, 2],
                mode='markers+text',
                marker=dict(
                    size=sizes_plot,
                    sizemode='diameter',
                    color=spos_plot,
                    colorscale=color_palette,
                    colorbar=dict(title='SPOs'),
                    opacity=0.8
                ),
                text=id_source_plot,
                textposition='top center',
                hovertemplate='<b>%{text}</b><br>'
                              'x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>'
                              'SPO: %{marker.color:.2f}<extra></extra>'
            )
        )
        fig.update_layout(
            title=title,
            plot_bgcolor='white',
            paper_bgcolor='white',
            scene=dict(
                xaxis=dict(
                    title='D1',
                    backgroundcolor="rgba(0, 0, 0, 0)",
                    gridcolor="lightgrey",
                    showbackground=True,
                    zerolinecolor="black",
                    titlefont=dict(color='black'),
                    tickfont=dict(color='black')
                ),
                yaxis=dict(
                    title='D2',
                    backgroundcolor="rgba(0, 0, 0, 0)",
                    gridcolor="lightgrey",
                    showbackground=True,
                    zerolinecolor="black",
                    titlefont=dict(color='black'),
                    tickfont=dict(color='black')
                ),
                zaxis=dict(
                    title='D3',
                    backgroundcolor="rgba(0, 0, 0,0)",
                    gridcolor="lightgrey",
                    showbackground=True,
                    zerolinecolor="black",
                    titlefont=dict(color='black'),
                    tickfont=dict(color='black')
                ),
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )

    return fig