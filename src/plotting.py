"""
Shared plotting utilities for visualizing discovered chains, so each
case study script does not need to duplicate this logic.
"""

import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_chain(series, chain, m, title=None, year_labels=False,
                start_year=2004, block_size=52):
    """
    Plots a time series with discovered chain subsequences highlighted.

    year_labels / start_year / block_size reproduce the Web Query case
    study's seasonal shading (Fig. 10 / Fig. 11 in the paper); omit
    year_labels for other case studies.
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(series, linewidth=1, color="black")

    for idx in chain:
        y = series[idx:idx + m]
        x = range(idx, idx + len(y))
        ax.plot(x, y, linewidth=3)

    if year_labels:
        color = itertools.cycle(["white", "gainsboro"])
        for i, x in enumerate(range(0, len(series), block_size)):
            ax.text(x + block_size * 0.25, max(series) * 0.9,
                     str(start_year + i), color="black", fontsize=12)
            rect = Rectangle((x, min(series)), block_size,
                              max(series) - min(series),
                              facecolor=next(color), alpha=0.5)
            ax.add_patch(rect)

    if title:
        ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    fig.tight_layout()
    return fig
