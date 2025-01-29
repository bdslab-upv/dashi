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
Main function for multi-batch metrics exploration.
"""

from typing import Dict

import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import get_colorscale

from .arrange_metrics import arrange_performance_metrics

_FONTSIZE = 14


def plot_multibatch_performance(*, metrics: Dict[str, float], metric_name: str) -> None:
    """
    Plots a heatmap visualizing the specified metric for multiple batches of training and test models.

    The function takes a dictionary of metrics and filters them based on the metric identifier.
    It then generates a heatmap where the x-axis represents the test batches,
    the y-axis represents the training batches, and the color scale indicates the
    values of the specified metric.

    The plot is interactive and can be explored (zoomed, hovered, etc.) using Plotly.

    Parameters
    ----------
    metrics : dict
        A dictionary where keys are tuples of (training_batch, test_batch, dataset_type),
        and values are the metric values for the corresponding combination.
        The `dataset_type` should be `'test'` to include the metric in the heatmap.

    metric_name : str
        The name of the metric to visualize.
        The function will filter metrics based on this identifier and only plot those for the 'test' set.
        Regression metric names, when applicable:
            - 'MEAN_ABSOLUTE_ERROR'
            - 'MEAN_SQUARED_ERROR'
            - 'ROOT_MEAN_SQUARED_ERROR'
            - 'R_SQUARED'
        Classification metric names, when applicable:
            - 'AUC_{class_identifier}'
            - 'AUC_MACRO'
            - 'LOGLOSS'
            - 'RECALL_{class_identifier}'
            - 'PRECISION_{class_identifier}'
            - 'F1-SCORE_{class_identifier}'
            - 'ACCURACY'
            - 'RECALL_MACRO'
            - 'RECALL_MICRO'
            - 'RECALL_WEIGHTED'
            - 'PRECISION_MACRO'
            - 'PRECISION_MICRO'
            - 'PRECISION_WEIGHTED'
            - 'F1-SCORE_MACRO'
            - 'F1-SCORE_MICRO'
            - 'F1-SCORE_WEIGHTED'

    Returns
    -------
    None
        This function generates and displays an interactive heatmap using Plotly,
        and does not return any value. The heatmap is displayed directly in the output
        environment (e.g., Jupyter notebook, web browser).

    Raises
    ------
    TypeError
        If the `metrics` parameter is not a dictionary or if `metric_identifier` is not a string.
    """

    # Metrics arrangement
    metrics_test_frame = arrange_performance_metrics(metrics=metrics, metric_name=metric_name)

    # Color scale definition
    colorscale = get_colorscale('RdYlGn')
    if metric_name in ('MEAN_ABSOLUTE_ERROR', 'MEAN_SQUARED_ERROR', 'ROOT_MEAN_SQUARED_ERROR', 'LOGLOSS'):
        colorscale = colorscale[::-1]

    # Plotting using Plotly
    heatmap_data = go.Heatmap(
        z=metrics_test_frame.values,  # Values for the heatmap (reversed rows)
        x=metrics_test_frame.columns,  # Columns as x-axis
        y=metrics_test_frame.index,  # Rows as y-axis
        colorscale=colorscale,  # Color scale
        colorbar=dict(title=metric_name),  # Colorbar label
        hovertemplate="%{y}<br>%{x}: %{z:.3f}",  # Tooltip on hover
        showscale=True  # Display colorbar scale
    )

    # Layout of the plot
    layout = go.Layout(
        title=f'{metric_name.lower().capitalize()} heatmap',
        xaxis=dict(title='Test Batch', tickangle=45, tickfont=dict(size=_FONTSIZE - 2)),
        yaxis=dict(title='Training Batch', tickfont=dict(size=_FONTSIZE - 2)),
        font=dict(size=_FONTSIZE, family="serif"),
        template="plotly_white"  # Optional: use a clean white background template
    )

    # Set the Plotly renderer for Jupyter or standalone use
    # pio.renderers.default = 'notebook'  # For Jupyter Notebooks (use 'notebook' or 'jupyterlab')
    # For standalone (non-Jupyter) use, you can also use:
    #pio.renderers.default = 'browser'

    # Create the figure and plot
    fig = go.Figure(data=[heatmap_data], layout=layout)
    fig.show()
