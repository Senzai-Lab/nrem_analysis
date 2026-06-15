import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
import seaborn as sns

def group_by_ids(
    values: np.ndarray,
    ids: np.ndarray,
    select_ids=None
    ) -> dict:
    assert values.ndim == 1, "values must be a 1D array"
    assert ids.ndim == 1, "ids must be a 1D array"
    assert values.shape[0] == ids.shape[0], "values and ids must have the same length"

    if select_ids is None:
        select_ids = np.unique(ids)
    return {uid: values[ids == uid] for uid in select_ids}

def load_tsg(
    values: np.ndarray,
    ids: np.ndarray,
    select_ids=None
    ) -> nap.TsGroup:
    grouped = group_by_ids(values, ids, select_ids)
    return nap.TsGroup({uid: nap.Ts(spikes, time_units='s') for uid, spikes in grouped.items()})

def plot_intervals(
    intervals: nap.IntervalSet,
    column: str = None,
    title: str = None,
    min_dur: float = 2,
    palette: str = 'deep',
    figsize = (14, 2.5),
    ax = None
    ):
    intervals = intervals.drop_short_intervals(min_dur)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if column is None:
        xranges = np.column_stack([intervals.start, intervals.end - intervals.start])
        ax.broken_barh(xranges, (0 - 0.4, 0.8), facecolors=sns.color_palette(palette, 1)[0], edgecolor='none')
        ax.set_yticks([0])
        ax.set_ylim(-0.5, 0.5)
    else:
        states = np.unique(intervals[column])
        n_states = len(states)
        colors = dict(zip(states, sns.color_palette(palette, n_states)))
        
        fig_height = figsize[1]
        bar_height = min(0.8, fig_height / (n_states + 1))  # Leave some padding

        for i, state in enumerate(states):
            epochs = intervals[intervals[column] == state]
            xranges = np.column_stack([epochs.start, epochs.end - epochs.start])
            ax.broken_barh(xranges, (i - bar_height/2, bar_height), facecolors=colors[state], edgecolor='none')
        
        ax.set_yticks(range(n_states), labels=states)
        ax.set_ylim(-0.5, n_states - 0.5)
    ax.set_xlabel('Time (s)')
    ax.set_xlim(intervals.start.min(), intervals.end.max())
    ax.spines[['top', 'right']].set_visible(False)
    
    if ax is None:
        plt.tight_layout()
    if title is not None:
        ax.set_title(title)
    return fig, ax