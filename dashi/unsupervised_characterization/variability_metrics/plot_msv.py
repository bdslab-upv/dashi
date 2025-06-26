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
from dashi.unsupervised_characterization.variability_metrics.estimate_msv_metrics import MSVMetrics
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from dashi._constants import VALID_COLOR_PALETTES

pio.renderers.default = "browser"

def plot_MSV(msv_metrics: MSVMetrics,
             dimensions: int=1,
             color_palette: str='Spectral'
             ) -> go.Figure:
    """
    Plots a Multi Source Variability (MSV) metrics visualization from a \code{MSVMetrics} object.

    Parameters
    ----------
    msv_metrics : MSVMetrics
        An instance of the `MSVMetrics` class containing the metrics to be plotted.

    dimensions : int, optional
        The number of dimensions for the plot. Must be 1, 2, or 3. Default is 1.

    color_palette : str, optional
        The color palette to use for the plot. Must be one of the valid color palettes defined in `VALID_COLOR_PALETTES`.
        Default is 'Spectral'.

    Returns
    -------
    go.Figure
        A Plotly figure object containing the MSV metrics visualization.
    """

    if dimensions not in [1, 2, 3]:
        raise ValueError('Dimensions must be 1, 2, or 3.')

    if dimensions >= len(msv_metrics.sources):
        raise ValueError(f'Dimensions must go from 1 to the number of sources - 1. Number of sources: '
                         f'{len(msv_metrics.sources)}')

    if color_palette not in VALID_COLOR_PALETTES:
        raise ValueError(f'Invalid color palette. Choose from: {VALID_COLOR_PALETTES}')

    vertices = msv_metrics.vertices
    spos = msv_metrics.SPO
    n_by_source = msv_metrics.nBySource
    id_source = msv_metrics.sources

    sphere_max_size = 100
    scale_factor = sphere_max_size / max(n_by_source)

    sizes = [int(x) * scale_factor for x in n_by_source]

    if dimensions == 1:
        fig = go.Figure(
            data=go.Scatter(
                x=vertices[:, 0],
                y=[0] * len(vertices),  # y-coordinates are zero for 1D
                mode='markers+text',
                marker=dict(
                    size=sizes,
                    sizemode='diameter',
                    color=spos,
                    colorscale=color_palette,
                    opacity=0.8,
                    colorbar=dict(title='SPOs')
                ),
                text=id_source,
                textposition='top center',
                hovertemplate='<b>%{text}</b><br>'
                              'x: %{x:.2f}<br>'
                              'SPO: %{marker.color:.2f}<extra></extra>'
            )
        )
        fig.update_layout(
            title='Multi Source Variability (MSV) Metrics',
            xaxis_title='D1',
            yaxis_title='D2',
            margin=dict(l=0, r=0, b=0, t=30),
            template='plotly_white',
        ),
        fig.show()
        return fig

    elif dimensions == 2:
        fig = go.Figure(
            data=go.Scatter(
                x=vertices[:, 0],
                y=vertices[:, 1],
                mode='markers+text',
                marker=dict(
                    size=sizes,
                    sizemode='diameter',
                    color=spos,
                    colorscale=color_palette,
                    opacity=0.8,
                    colorbar=dict(title='SPOs')
                ),
                text=id_source,
                textposition='top center',
                hovertemplate='<b>%{text}</b><br>'
                              'x: %{x:.2f}<br>y: %{y:.2f}<br>'
                              'SPO: %{marker.color:.2f}<extra></extra>'
            )
        )
        fig.update_layout(
            title='Multi Source Variability (MSV) Metrics',
            xaxis_title='D1',
            yaxis_title='D2',
            margin=dict(l=0, r=0, b=0, t=30),
            template='plotly_white',
        ),
        fig.show()
        return fig

    elif dimensions == 3:
        fig = go.Figure(
            data=go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode='markers+text',
                marker=dict(
                    size=sizes,
                    sizemode='diameter',
                    color=spos,
                    colorscale=color_palette,
                    colorbar=dict(title='SPOs'),
                    opacity=0.8
                ),
                text=id_source,
                textposition='top center',
                hovertemplate='<b>%{text}</b><br>'
                              'x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>'
                              'SPO: %{marker.color:.2f}<extra></extra>'
            )
        )
        fig.update_layout(
            title="Multi-Source Variability plot",
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
        ),
        fig.show()
        return fig
