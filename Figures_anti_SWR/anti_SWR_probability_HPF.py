import matplotlib.pyplot as plt
import seaborn as sns
from Utils.Settings import window_spike_hist, output_folder_calculations, neuropixel_dataset, var_thr, minimum_ripples_count_spike_analysis, minimum_ripples_count_generated_in_lateral_or_medial_spike_analysis
import dill
from Utils.Utils import acronym_color_map
import pandas as pd

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

anti_SWR_counts_brain_regions = pd.Series(subset_clusters[subset_clusters["Parent brain region"]=="HPF"].groupby(["Session id", "Brain region"])\
.apply(lambda x: x[x['Ripple modulation (0-50 ms)']<-.5].shape[0]/x.shape[0]), name="Probability anti-SWR")
anti_SWR_counts_brain_regions = pd.DataFrame(anti_SWR_counts_brain_regions).reset_index()

fig, axs = plt.subplots(2)
sns.barplot(data=anti_SWR_counts_brain_regions, y="Probability anti-SWR", x="Brain region", palette=palette_areas, ax=axs[0])
sns.barplot(data=pd.Series(subset_clusters[(subset_clusters["Parent brain region"]=="HPF")&(subset_clusters["Anti-SWR"]==True)].groupby(["Session id", "Brain region"]).size(), name="Number anti-SWR neurons").reset_index(),
            y="Number anti-SWR neurons", x="Brain region", palette=palette_areas, ax=axs[1])
axs[1].set_title("Number anti-SWR neurons per session")
plt.show()