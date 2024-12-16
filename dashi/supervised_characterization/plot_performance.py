"""
DESCRIPTION: main function for multi-batch metrics exploration.
AUTHOR: Pablo Ferri-Borredà
DATE: 17/10/24
"""

# MODULES IMPORT
from typing import Dict

import plotly.graph_objects as go
import plotly.io as pio

from .arrange_metrics import arrange_performance_metrics

# SETTINGS
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
        The name of the metric to visualize (e.g., 'accuracy', 'loss', etc.).
        The function will filter metrics based on this identifier and only plot those for the 'test' set.

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

    # Plotting using Plotly
    heatmap_data = go.Heatmap(
        z=metrics_test_frame.values,  # Values for the heatmap (reversed rows)
        x=metrics_test_frame.columns,  # Columns as x-axis
        y=metrics_test_frame.index,  # Rows as y-axis
        colorscale='RdYlGn',  # Color scale
        colorbar=dict(title=metric_name),  # Colorbar label
        zmid=0.5,  # Center color (0 value centered)
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
    pio.renderers.default = 'browser'

    # Create the figure and plot
    fig = go.Figure(data=[heatmap_data], layout=layout)
    fig.show()
