"""
DESCRIPTION: main function for results exploration.
AUTHOR: Pablo Ferri-Borredà
DATE: 17/10/24
"""

# MODULES IMPORT
import tkinter as tk

import matplotlib as mpl_

mpl_.use('TkAgg')
from typing import Dict
import matplotlib.pyplot as mpl
from seaborn import heatmap
from pandas import Series

# SETTINGS
FONTSIZE = 14


# FUNCTION DEFINITION
def plot_multi_batch_models(*, metrics: Dict[str, float], metric_identifier: str) -> None:
    # Inputs checking
    if type(metrics) is not dict:
        raise TypeError('Metrics should be specified in a dictionary.')
    if type(metric_identifier) is not str:
        raise TypeError('Metric identifier needs to be specified as a string.')

    # Selection of the metrics relative to the test set
    metrics_test = {(combination[0], combination[1]): metrics_[metric_identifier] for combination, metrics_ in
                    metrics.items() if combination[2] == 'test'}

    # Data formatting
    metrics_test_frame = Series(metrics_test).unstack()

    # Plotting
    root = tk.Tk()
    mpl.rc('font', family='serif')
    mpl.figure()

    heatmap_plot = heatmap(
        metrics_test_frame.iloc[::-1], annot=True, fmt=".3f", cmap='RdYlGn', annot_kws={'size': FONTSIZE - 2}
    )

    cax = heatmap_plot.figure.axes[-1]
    cax.tick_params(labelsize=FONTSIZE - 2)

    mpl.xticks(fontsize=FONTSIZE - 2, fontname='serif', rotation=45)
    mpl.xlabel('Test Batch', fontsize=FONTSIZE)

    mpl.yticks(rotation=0, fontsize=FONTSIZE - 2, fontname='serif')
    mpl.ylabel('Training Batch', fontsize=FONTSIZE)
    mpl.title(f'{metric_identifier.lower().capitalize()} heatmap', fontsize=FONTSIZE + 2)

    mpl.show()
    root.mainloop()
