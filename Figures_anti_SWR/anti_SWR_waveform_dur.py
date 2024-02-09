import matplotlib.pyplot as plt
import seaborn as sns
from Utils.Settings import window_spike_hist, output_folder_calculations, neuropixel_dataset, var_thr, minimum_ripples_count_spike_analysis, minimum_ripples_count_generated_in_lateral_or_medial_spike_analysis
import dill
from Utils.Utils import acronym_color_map

palette_type_neuron={"Putative exc": "#D64933", "Putative inh": "#00C2D1"}
palette_anti_SWR={True: "#A268E4", False: "#454545"}

with open(f"{output_folder_calculations}/subset_clusters.pkl", 'rb') as f:
    subset_clusters = dill.load(f)

with open(f"{output_folder_calculations}/anti_SWR_counts.pkl", 'rb') as f:
    anti_SWR_counts = dill.load(f)

fig, axs = plt.subplots(1, 4, figsize=(20,5))
sns.kdeplot(palette=palette_anti_SWR, data=subset_clusters[(subset_clusters["Parent brain region"]=="HPF")], x="Waveform duration", hue="Anti-SWR", fill=True, common_norm=False, ax=axs[1])
sns.histplot(palette=palette_anti_SWR,  data=subset_clusters[(subset_clusters["Parent brain region"]=="HPF")], x="Waveform duration",  ax=axs[2], hue="Anti-SWR", cumulative=True,
             element="step", fill=False, stat="density", common_norm=False)
axs[2].axvline(.4, linestyle="--", alpha=.5, color=".15")
sns.boxenplot(palette=palette_anti_SWR, data=subset_clusters[(subset_clusters["Parent brain region"]=="HPF")&(subset_clusters["Waveform duration"]>=0.4)], y="Waveform duration", x="Anti-SWR", ax= axs[3])
axs[3].set_title("Waveform duration putative exc")

y = "Waveform amplitude"
sns.kdeplot(ax= axs[0], data=subset_clusters[(subset_clusters["Parent brain region"]=="HPF")], x="Waveform duration", hue="Type neuron", y=y, palette=palette_type_neuron, fill=True, alpha=.5)
sns.scatterplot(color=palette_anti_SWR[True], ax= axs[0], data=subset_clusters[((subset_clusters["Parent brain region"]=="HPF"))&(subset_clusters["Anti-SWR"]==True)], x="Waveform duration", y=y, s=6, alpha=.6)
axs[0].axvline(.4, linestyle="--", alpha=.5, color=".15")

plt.show()