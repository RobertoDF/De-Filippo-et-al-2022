import matplotlib.pyplot as plt
import seaborn as sns
from Utils.Settings import window_spike_hist, output_folder_calculations, neuropixel_dataset, var_thr, minimum_ripples_count_spike_analysis, minimum_ripples_count_generated_in_lateral_or_medial_spike_analysis
import dill
from Utils.Utils import acronym_color_map

with open(f"{output_folder_calculations}/subset_clusters.pkl", 'rb') as f:
    subset_clusters = dill.load(f)

palette_parent_areas = dict()
for area in subset_clusters['Parent brain region'].unique():
    palette_parent_areas[area] = '#' + ''.join(f'{i:02X}' for i in acronym_color_map[area]);

palette_areas = dict()
for area in subset_clusters['Brain region'].unique():
    palette_areas[area] = '#' + ''.join(f'{i:02X}' for i in acronym_color_map[area]);

g = sns.FacetGrid(subset_clusters, col="Parent brain region", hue="Parent brain region", palette=palette_parent_areas)
g.map(sns.histplot, "Ripple modulation (0-50 ms)", element="poly", stat="probability")
g.set(xlim=(-1, 6))

# flatten axes into a 1-d array
axes = g.axes.flatten()

# iterate through the axes
for ax in axes:
    ax.axvline(.5, color=".15", linestyle="--")
    ax.axvline(-.5, color=".15", linestyle="--")

plt.show()