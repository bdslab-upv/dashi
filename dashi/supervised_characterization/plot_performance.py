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

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import warnings
import plotly.graph_objects as go
from plotly.colors import get_colorscale
from sklearn.cluster import SpectralBiclustering

from .arrange_metrics import arrange_performance_metrics

__all__ = ['plot_multibatch_performance']

_FONTSIZE = 14


def _infer_batching_type(
    *,
    row_labels: Sequence,
    col_labels: Sequence,
    threshold: float = 0.8
) -> str:
    """
    Helper for infering batching type from axis labels using datetime-parseability.

    Heuristic:
    - If >= threshold of labels can be parsed as datetime on either axis -> 'date'
    - Otherwise -> 'source'
    """

    def _datetime_ratio(labels: Sequence) -> float:
        if len(labels) == 0:
            return 0.0

        s = pd.Series(labels, dtype="object").astype(str).str.strip()

        # month period
        parsed = pd.to_datetime(s, format="%B %Y", errors="coerce")

        # year period
        missing = parsed.isna()
        if missing.any():
            parsed.loc[missing] = pd.to_datetime(s.loc[missing], format="%Y", errors="coerce")

        # generic fallback
        missing = parsed.isna()
        if missing.any():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Could not infer format, so each element will be parsed individually.*",
                    category=UserWarning,
                )
                parsed.loc[missing] = pd.to_datetime(s.loc[missing], errors="coerce")

        return float(parsed.notna().mean())

    row_ratio = _datetime_ratio(row_labels)
    col_ratio = _datetime_ratio(col_labels)

    if max(row_ratio, col_ratio) >= threshold:
        return 'date'
    return 'source'


def _bicluster_reorder(
    *,
    frame: pd.DataFrame,
    n_clusters: int,
    random_state: int
) -> pd.DataFrame:
    """
    Reorder a matrix with spectral biclustering.

    Returns a reordered DataFrame preserving original labels (reordered).
    """
    n_rows, n_cols = frame.shape
    if n_rows < 3 or n_cols < 3:
        return frame

    n_clusters = max(2, min(n_clusters, n_rows, n_cols))

    values = frame.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        finite_mask = np.isfinite(values)
        fill_value = float(values[finite_mask].mean()) if finite_mask.any() else 0.0
        values[~finite_mask] = fill_value

    model = SpectralBiclustering(
        n_clusters=(n_clusters, n_clusters),
        random_state=random_state
    )
    model.fit(values)

    row_order = np.argsort(model.row_labels_)
    col_order = np.argsort(model.column_labels_)

    reordered = frame.iloc[row_order, :].iloc[:, col_order]
    return reordered


def plot_multibatch_performance(
    *,
    metrics: Dict[str, float],
    metric_name: str,
    batching_type: Optional[str] = None,
    apply_biclustering_on_source: bool = True,
    n_clusters: int = 3,
    random_state: int = 42,
) -> go.Figure:
    """
    Plots a heatmap visualizing the specified metric for multiple batches of training and test models.

    The function takes a dictionary of metrics and filters them based on the metric identifier.
    It then generates a heatmap where the x-axis represents the test batches,
    the y-axis represents the training batches, and the color scale indicates the
    values of the specified metric.

    If batching_type is not provided, it is inferred from labels:
      - date-batched: keep natural order (chronological interpretability)
      - source-batched: optionally reorder matrix with biclustering for block structure visibility

    Parameters
    ----------
    metrics : dict
        A dictionary where keys are tuples of (training_batch, test_batch, dataset_type),
        and values are the metric values for the corresponding combination.
        The `dataset_type` should be `'test'` to include the metric in the heatmap.

    metric_name : str
        The name of the metric to visualize.

    batching_type : Optional[str], default=None
        Explicit batching mode: 'date' or 'source'.
        If None, inferred from axis labels.

    apply_biclustering_on_source : bool, default=True
        If True, applies biclustering reordering when batching is source-based.

    n_clusters : int, default=3
        Number of row/column clusters for spectral biclustering.

    random_state : int, default=42
        Seed for deterministic biclustering ordering.

    Returns
    -------
    fig
        A Plotly figure object containing the heatmap visualization of the specified metric.
    """

    # Metrics arrangement
    metrics_test_frame = arrange_performance_metrics(metrics=metrics, metric_name=metric_name)

    if batching_type is None:
        detected_batching_type = _infer_batching_type(
            row_labels=metrics_test_frame.index.tolist(),
            col_labels=metrics_test_frame.columns.tolist(),
        )
    else:
        detected_batching_type = batching_type.lower().strip()
        if detected_batching_type not in ('date', 'source'):
            raise ValueError("batching_type must be either 'date' or 'source'.")

    # Apply biclustering only for source batching
    if detected_batching_type == 'source' and apply_biclustering_on_source:
        metrics_test_frame = _bicluster_reorder(
            frame=metrics_test_frame,
            n_clusters=n_clusters,
            random_state=random_state,
        )

    # Color scale definition
    colorscale = get_colorscale('RdYlGn')
    if metric_name in ('MEAN_ABSOLUTE_ERROR', 'MEAN_SQUARED_ERROR', 'ROOT_MEAN_SQUARED_ERROR', 'LOGLOSS'):
        colorscale = colorscale[::-1]

    # Plotting using Plotly
    heatmap_data = go.Heatmap(
        z=metrics_test_frame.values,
        x=metrics_test_frame.columns,
        y=metrics_test_frame.index,
        colorscale=colorscale,
        colorbar=dict(title=metric_name),
        hovertemplate="%{y}<br>%{x}: %{z:.3f}",
        showscale=True
    )

    # Layout of the plot
    layout = go.Layout(
        title=f'{metric_name.lower().capitalize()} heatmap',
        xaxis=dict(title='Test Batch', tickangle=45, tickfont=dict(size=_FONTSIZE - 2)),
        yaxis=dict(title='Training Batch', tickfont=dict(size=_FONTSIZE - 2)),
        font=dict(size=_FONTSIZE, family="serif"),
        template="plotly_white"
    )

    # Create the figure and plot
    fig = go.Figure(data=[heatmap_data], layout=layout)

    return fig
