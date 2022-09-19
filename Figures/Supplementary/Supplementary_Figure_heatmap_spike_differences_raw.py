import dill
import matplotlib.pyplot as plt
from Utils.Settings import output_folder_calculations, output_folder_figures_calculations, Adapt_for_Nature_style
from Utils.Utils import Naturize
from Utils.Utils import postprocess_spike_hists
import numpy as np

with open(f"{output_folder_calculations}/spike_hists_summary_per_parent_area_by_seed_location.pkl", 'rb') as f:
    spike_hists = dill.load(f)

with open(f"{output_folder_figures_calculations}/temp_data_figure_4.pkl", 'rb') as f:
    space_sub_spike_times, target_area, units, field_to_use_to_compare, \
    session_id_example, lfp, lfp_per_probe, \
    ripple_cluster_lateral_seed, ripple_cluster_medial_seed, source_area, ripples, \
    tot_summary_early, summary_fraction_active_clusters_per_ripples_early, \
    summary_fraction_active_clusters_per_ripples_early_by_neuron_type, \
    tot_summary_late, summary_fraction_active_clusters_per_ripples_late, \
    summary_fraction_active_clusters_per_ripples_late_by_neuron_type, \
    tot_summary, summary_fraction_active_clusters_per_ripples, \
    summary_fraction_active_clusters_per_ripples_by_neuron_type = dill.load(f)


lrs, out_hist_medial, out_hist_lateral = spike_hists[session_id_example, 'HPF', 'central']

means_cut = postprocess_spike_hists(out_hist_lateral, out_hist_medial, lrs)

fig, axs = plt.subplots(1)
ax = means_cut.diff("Seed").squeeze().plot.pcolormesh(cmap="seismic", add_colorbar=False)
axs.invert_yaxis()
axs.vlines(x=0, ymin=axs.get_ylim()[0], ymax=axs.get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs.vlines(x=50, ymin=axs.get_ylim()[0], ymax=axs.get_ylim()[1],  colors='white', ls='--', linewidth=1)

plt.xlabel("Time from ripple start (ms)")
plt.ylabel("M-L (µm)")
cbar = plt.colorbar(ax, shrink=.5)
cbar.ax.set_ylabel('Δ spiking per 10 ms', rotation=270, labelpad=8)
cbar.ax.set_label("colorbar_example") # needed to distinguish colobars

for d in ["left", "top", "bottom", "right"]:
    plt.gca().spines[d].set_visible(False)

axs.set_title("Medial seed - Lateral seed", fontsize=7)
plt.show()
