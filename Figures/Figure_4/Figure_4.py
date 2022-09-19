import matplotlib.pyplot as plt
import pylustrator
import dill
from Utils.Style import palette_ML
from Utils.Settings import output_folder, output_folder_figures_calculations, Adapt_for_Nature_style
from Utils.Utils import Naturize

with open(f"{output_folder_figures_calculations}/temp_data_figure_4.pkl", 'rb') as f:
    space_sub_spike_times, target_area, units, field_to_use_to_compare,\
    session_id_example, lfp, lfp_per_probe,\
    ripple_cluster_lateral_seed, ripple_cluster_medial_seed, source_area, ripples,\
    tot_summary_early, summary_fraction_active_clusters_per_ripples_early, \
    summary_fraction_active_clusters_per_ripples_early_by_neuron_type,\
    tot_summary_late, summary_fraction_active_clusters_per_ripples_late,\
    summary_fraction_active_clusters_per_ripples_late_by_neuron_type,\
tot_summary, summary_fraction_active_clusters_per_ripples, \
summary_fraction_active_clusters_per_ripples_by_neuron_type = dill.load(f)

pylustrator.start()

pylustrator.load(f"{output_folder_figures_calculations}/Figure_4_brainrender_crop.png")
pylustrator.load("Figure_4_example_medial_seed_ripple.py", offset=[0.25, 0])
pylustrator.load("Figure_4_example_lateral_seed_ripple.py", offset=[0.49, 0])
pylustrator.load("Figure_4_example_summary_hists_per_ML.py", offset=[0.72, 0])
pylustrator.load("Figure_4_heatmap_spike_differences_interp.py")
pylustrator.load("Figure_4_summary_heatmap_spike_differences_interp.py")
pylustrator.load("Figure_4_duration_ML_by_ripple_strength.py")
pylustrator.load("Figure_4_clusters_per_ripple_early_late.py")
pylustrator.load("Figure_4_spiking_rate_early_late.py")

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(17.940000/2.54, 18.830000/2.54, forward=True)
plt.figure(1).ax_dict[f"{output_folder_figures_calculations}/Figure_4_brainrender_crop.png"].set_position([-0.077024, 0.664205, 0.453828, 0.262319])
plt.figure(1).ax_dict["/alzheimer/Roberto/Allen_Institute/temp/Figure_4_brainrender_crop.png"].set_position([-0.055388, 0.664205, 0.410557, 0.262319])
plt.figure(1).ax_dict["colorbar_example"].set_position([0.287960, 0.383508, 0.004811, 0.091667])
plt.figure(1).ax_dict["colorbar_summary"].set_position([0.629875, 0.383258, 0.004824, 0.091917])
plt.figure(1).axes[1].set_position([0.283278, 0.915118, 0.206323, 0.037263])
plt.figure(1).axes[15].set_position([0.757750, 0.942530, 0.231151, 0.049284])
plt.figure(1).axes[16].set_position([0.757750, 0.883389, 0.231151, 0.049284])
plt.figure(1).axes[17].set_position([0.757750, 0.824248, 0.231151, 0.049284])
plt.figure(1).axes[18].set_position([0.757750, 0.765107, 0.231151, 0.049284])
plt.figure(1).axes[19].set_position([0.757750, 0.705966, 0.231151, 0.049284])
plt.figure(1).axes[20].legend(borderpad=0.2, labelspacing=0.2, handletextpad=0.6, title="Seed")
plt.figure(1).axes[20].set_position([0.757750, 0.646825, 0.231151, 0.049284])
plt.figure(1).axes[20].get_legend()._set_loc((-0.061453, 6.223354))
plt.figure(1).axes[21].set_position([0.057977, 0.333204, 0.223060, 0.223150])
plt.figure(1).axes[21].title.set_fontsize(5)
plt.figure(1).axes[23].set_position([0.398465, 0.333204, 0.223060, 0.223150])
plt.figure(1).axes[23].title.set_fontsize(5)
plt.figure(1).axes[25].set_position([0.738953, 0.333204, 0.249948, 0.262319])
plt.figure(1).axes[25].get_legend()._set_loc((0.436757, 0.918619))
plt.figure(1).axes[26].set_position([0.057977, 0.048338, 0.167295, 0.192284])
plt.figure(1).axes[27].set_position([0.312520, 0.048338, 0.167295, 0.192284])
plt.figure(1).axes[28].set_position([0.567063, 0.048338, 0.167295, 0.192284])
plt.figure(1).axes[29].set_position([0.821606, 0.048338, 0.167295, 0.192284])
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.009915, 0.983516])
plt.figure(1).texts[0].set_text("A")
plt.figure(1).texts[0].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.253541, 0.983516])
plt.figure(1).texts[1].set_text("B")
plt.figure(1).texts[1].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.729462, 0.983516])
plt.figure(1).texts[2].set_text("C")
plt.figure(1).texts[2].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_fontsize(8)
plt.figure(1).texts[3].set_ha("center")
plt.figure(1).texts[3].set_position([0.149890, 0.966422])
plt.figure(1).texts[3].set_text(f"Session {session_id_example}")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_fontsize(7)
plt.figure(1).texts[4].set_ha("center")
plt.figure(1).texts[4].set_position([0.385636, 0.966422])
plt.figure(1).texts[4].set_text("Medial seed")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_fontsize(7)
plt.figure(1).texts[5].set_position([0.583548, 0.966422])
plt.figure(1).texts[5].set_text("Lateral seed")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_position([0.009915, 0.585149])
plt.figure(1).texts[6].set_text("D")
plt.figure(1).texts[6].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[7].new
plt.figure(1).texts[7].set_position([0.338527, 0.585149])
plt.figure(1).texts[7].set_text("E")
plt.figure(1).texts[7].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[8].new
plt.figure(1).texts[8].set_fontsize(7)
plt.figure(1).texts[8].set_position([0.104816, 0.588812])
plt.figure(1).texts[8].set_text(f"Session {session_id_example}")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[9].new
plt.figure(1).texts[9].set_fontsize(7)
plt.figure(1).texts[9].set_ha("center")
plt.figure(1).texts[9].set_position([0.512415, 0.588812])
plt.figure(1).texts[9].set_text("Grand average")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[10].new
plt.figure(1).texts[10].set_position([0.681303, 0.585149])
plt.figure(1).texts[10].set_text("F")
plt.figure(1).texts[10].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[11].new
plt.figure(1).texts[11].set_position([0.009915, 0.264861])
plt.figure(1).texts[11].set_text("G")
plt.figure(1).texts[11].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[12].new
plt.figure(1).texts[12].set_position([0.484419, 0.264861])
plt.figure(1).texts[12].set_text("H")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure, color=palette_ML["Central"] )  # id=plt.figure(1).texts[13].new
plt.figure(1).texts[12].set_weight("bold")
plt.figure(1).texts[13].set_fontsize(7)
plt.figure(1).texts[13].set_ha("center")
plt.figure(1).texts[13].set_position([0.152337, 0.943718])
plt.figure(1).texts[13].set_text("Central reference")

if Adapt_for_Nature_style is True:
    Naturize()

#% end: automatic generated code from pylustrator
#plt.show()

plt.savefig(f"{output_folder}/Figure_4", dpi=300)