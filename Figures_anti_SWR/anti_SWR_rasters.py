import matplotlib.pyplot as plt
import numpy as np
from Utils.Settings import window_spike_hist, output_folder_calculations, neuropixel_dataset, var_thr, minimum_ripples_count_spike_analysis, minimum_ripples_count_generated_in_lateral_or_medial_spike_analysis
import dill
from Utils.Utils import acronym_color_map
import pandas as pd
from Utils.Settings import output_folder

with open(f'{output_folder_calculations}/anti_swr_spike_times.pkl', 'rb') as f:
    spikes_per_ripple = dill.load(f)

from itertools import cycle
from matplotlib.colors import ListedColormap
# create a ListedColormap from the "Set1" colormap
cmap = ListedColormap(plt.cm.Set2.colors)

# create a cyclic iterator over the colormap
colors = cycle(cmap.colors)

# Create a figure with subplots
n = 30
fig, axes = plt.subplots(nrows=len(list(spikes_per_ripple.keys())[:n]), sharex=True, figsize=(10, 10))

# Loop through each key in the dictionary
for i, key in enumerate(list(spikes_per_ripple.keys())[:n]):
    # Get the spikes for the current key
    spikes = spikes_per_ripple[key][0]

    # Plot the eventplot with the appropriate color and lineoffset
    axes[i].eventplot(spikes, linewidths=2, linelengths=5, color=next(colors))
    axes[i].axvline(0, linestyle="--", alpha=.5, color=".15")
    if i != n - 1:
        axes[i].axis('off')
    else:
        axes[i].set(yticklabels=[])  # remove the tick labels
        axes[i].spines['left'].set_visible(False)
        axes[i].set_xlabel("Time centered on ripple onset(s)")

# Set the yticks and labels
yticks = np.arange(len(spikes_per_ripple))

#plt.show()
plt.savefig(f"{output_folder}/rasters", dpi=300)