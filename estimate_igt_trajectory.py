from all_classes import IGTProjection
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


def estimate_igt_trajectory(igt_projection: IGTProjection, number_of_points=None):
    if igt_projection is None:
        raise ValueError("An input IGT projection object is required.")
    if number_of_points is None:
        number_of_points = len(igt_projection.projection) * 10

    dimensions = len(igt_projection.projection.columns)
    batches = len(igt_projection.projection)

    t = np.arange(1, batches + 1)
    tt = np.linspace(1, batches, num=number_of_points)

    trajectory_function = {}
    points = pd.DataFrame(np.zeros((number_of_points, dimensions)), columns=[f"D{i + 1}" for i in range(dimensions)])
    dates = np.linspace(min(igt_projection.data_temporal_map.dates),
                        max(igt_projection.data_temporal_map.dates), num=number_of_points)

    for i in range(dimensions):
        spline = UnivariateSpline(t, igt_projection.projection.iloc[:, i])
        trajectory_function[f"D{i + 1}"] = spline
        points.iloc[:, i] = spline(tt)

    results = {"points": points, "dates": dates, "trajectoryFunction": trajectory_function}
    return results
