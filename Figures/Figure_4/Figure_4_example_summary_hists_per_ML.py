import dill
from Utils.Settings import output_folder_calculations, output_folder_figures_calculations
from Utils.Utils import postprocess_spike_hists, plot_spike_hists_per_ML
import matplotlib.pyplot as plt

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

plot_spike_hists_per_ML(means_cut)
plt.show()

