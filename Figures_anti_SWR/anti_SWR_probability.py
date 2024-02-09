import matplotlib.pyplot as plt
import seaborn as sns
from Utils.Settings import window_spike_hist, output_folder_calculations, neuropixel_dataset, var_thr, minimum_ripples_count_spike_analysis, minimum_ripples_count_generated_in_lateral_or_medial_spike_analysis
import dill
from Utils.Utils import acronym_color_map

with open(f"{output_folder_calculations}/subset_clusters.pkl", 'rb') as f:
    subset_clusters = dill.load(f)

with open(f"{output_folder_calculations}/anti_SWR_counts.pkl", 'rb') as f:
    anti_SWR_counts = dill.load(f)

palette_parent_areas = dict()
for area in subset_clusters['Parent brain region'].unique():
    palette_parent_areas[area] = '#' + ''.join(f'{i:02X}' for i in acronym_color_map[area]);

palette_areas = dict()
for area in subset_clusters['Brain region'].unique():
    palette_areas[area] = '#' + ''.join(f'{i:02X}' for i in acronym_color_map[area]);



sns.stripplot(data=anti_SWR_counts, y="Probability anti-SWR", x="Parent brain region", size=4, color=".3",
              linewidth=1, palette=palette_parent_areas)
sns.boxplot(data=anti_SWR_counts, y="Probability anti-SWR", x="Parent brain region",  whis=[0, 100], width=.6,
            palette=palette_parent_areas)

plt.show()