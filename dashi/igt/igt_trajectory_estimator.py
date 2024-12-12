import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from dashi.igt.igt_projection import IGTProjection


def _estimate_igt_trajectory(igt_projection: IGTProjection, number_of_points=None):
    """
    Estimates a trajectory of the information temporal evolution in a IGT projection by fitting a
    cubic smoothing spline
    """
    if igt_projection is None:
        raise ValueError("An input IGT projection object is required.")
    if number_of_points is None:
        number_of_points = igt_projection.projection.shape[0] * 10

    dimensions = igt_projection.projection.shape[1]
    batches = igt_projection.projection.shape[0]

    t = np.arange(1, batches + 1)
    tt = np.linspace(1, batches, num=number_of_points)

    trajectory_function = {}
    points = pd.DataFrame(
        np.zeros((number_of_points, dimensions)),
        columns=[f"D{i + 1}" for i in range(dimensions)]
    )
    dates = pd.date_range(
        start=min(igt_projection.data_temporal_map.dates),
        end=max(igt_projection.data_temporal_map.dates),
        periods=number_of_points
    )

    for i in range(dimensions):
        spline = UnivariateSpline(x=t, y=igt_projection.projection[:, i], s=0.008, k=3)
        trajectory_function[f"D{i + 1}"] = spline
        points.iloc[:, i] = spline(tt)

    results = {"points": points, "dates": dates, "trajectoryFunction": trajectory_function}
    return results
