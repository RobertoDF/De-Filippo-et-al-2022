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

fig, axs = plt.subplots( 1 ,figsize=(15,8))


sns.histplot(data=subset_clusters, x='Ripple modulation (0-50 ms)',
                hue='Parent brain region', palette=palette_parent_areas, ax=axs, multiple="stack", element="poly", stat="probability")
axs.set_xlim((-1,4))
axs.axvline(-.5, color=".15", linestyle="--")
axs.axvline(.5, color=".15", linestyle="--")

plt.show()