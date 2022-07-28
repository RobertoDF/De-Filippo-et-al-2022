import matplotlib.pyplot as plt
import pylustrator
import pickle
from Utils.Settings import output_folder_supplementary , output_folder_figures_calculations

with open(f"{output_folder_figures_calculations}/temp_data_supp_figure_1.pkl", "rb") as fp:  # Unpickling
    _, _, session_id, _, _, _ = pickle.load(fp)

pylustrator.start()

pylustrator.load(f"{output_folder_figures_calculations}/probes_{session_id}_crop.png")
pylustrator.load("Figure_1_heatmaps.py", offset=[0.7, 0])
pylustrator.load("Figure_1_traces.py", offset=[0.27, 0.02])
pylustrator.load("Figure_1_scatterplot _ripple_lfp_areas.py", offset=[0.25, 0.2])
pylustrator.load("Figure_1_scatterplot _ripple_lfp_areas_summary.py", offset=[0.55, 0.2])
pylustrator.load("Figure_1_violinplot_summary.py", offset=[0.15, 0.2])

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(17.980000/2.54, 13.000000/2.54, forward=True)
plt.figure(1).ax_dict[f"{output_folder_figures_calculations}/probes_{session_id}_crop.png"].set_position([0.015432, 0.756061, 0.250000, 0.235326])
plt.figure(1).axes[1].set_xlim(0.0, 2895.0)
plt.figure(1).axes[1].set_position([0.664071, 0.429617, 0.122846, 0.530230])
plt.figure(1).axes[2].set_position([0.837367, 0.429599, 0.122846, 0.530230])
plt.figure(1).axes[3].set_position([0.968940, 0.640279, 0.009534, 0.135271])
plt.figure(1).axes[4].set_position([0.796856, 0.640279, 0.009534, 0.135271])
plt.figure(1).axes[6].title.set_position([0.500000, 1.044153])
plt.figure(1).axes[7].title.set_position([0.500000, 1.044153])
plt.figure(1).axes[29].set_position([0.041472, 0.441295, 0.208528, 0.294246])
plt.figure(1).axes[29].get_legend()._set_loc((0.461708, -0.015817))
plt.figure(1).axes[30].set_position([0.189109, 0.619071, 0.060891, 0.135271])
plt.figure(1).axes[31].set_xlim(-0.792928276530378, 16.65149380713794)
plt.figure(1).axes[31].set_xticks([0.0, 5.0, 10.0, 15.0])
plt.figure(1).axes[31].set_xticklabels(["0", "5", "10", "15"], fontsize=5.0, fontweight="normal", color=".15", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="center")
plt.figure(1).axes[31].set_position([0.041472, 0.079522, 0.208528, 0.294246])
plt.figure(1).axes[31].get_legend()._set_loc((0.340230, -0.005097))
plt.figure(1).axes[32].set_position([0.189109, 0.251753, 0.060891, 0.135076])
plt.figure(1).axes[33].set_position([0.297855, 0.079522, 0.670491, 0.296179])
plt.figure(1).axes[33].xaxis.labelpad = -3.441947
plt.figure(1).texts[0].set_position([0.816554, 0.991034])
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.002990, 0.965753])
plt.figure(1).texts[1].set_text("A")
plt.figure(1).texts[1].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.255583, 0.965753])
plt.figure(1).texts[2].set_text("B")
plt.figure(1).texts[2].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_position([0.598015, 0.965753])
plt.figure(1).texts[3].set_text("C")
plt.figure(1).texts[3].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_position([0.002990, 0.744406])
plt.figure(1).texts[4].set_text("D")
plt.figure(1).texts[4].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_position([0.002990, 0.382583])
plt.figure(1).texts[5].set_text("E")
plt.figure(1).texts[5].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_position([0.255583, 0.382583])
plt.figure(1).texts[6].set_text("F")
plt.figure(1).texts[6].set_weight("bold")
#% end: automatic generated code from pylustrator
#plt.show()
plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_2", dpi=300)
