import matplotlib.pyplot as plt
import pylustrator
from Utils.Settings import output_folder, output_folder_figures_calculations, Adapt_for_Nature_style
from Utils.Utils import Naturize
from Utils.Style import palette_ML
import dill

with open(f"{output_folder_figures_calculations}/temp_data_figure_2.pkl", 'rb') as f:
    session_id, session_trajs, columns_to_keep, ripples, real_ripple_summary,\
    lfp_per_probe, ripple_cluster_strong, ripple_cluster_weak, example_session = dill.load(f)


pylustrator.start()

pylustrator.load("Figure_2_traces_strong.py", offset=[0, 0])
pylustrator.load(f"{output_folder_figures_calculations}/Figure_2_brainrender_crop.png")
pylustrator.load("Figure_2_traces_weak.py")
pylustrator.load("Figure_2_session_line_averages.py")
pylustrator.load("Figure_2_summary_medial.py")
pylustrator.load("Figure_2_summary_lateral.py")
pylustrator.load("Figure_2_summary_center.py")
pylustrator.load(f"{output_folder_figures_calculations}/Figure_2_brainrender_medial_crop.png")
pylustrator.load(f"{output_folder_figures_calculations}/Figure_2_brainrender_lateral_crop.png")
pylustrator.load(f"{output_folder_figures_calculations}/Figure_2_brainrender_center_crop.png")


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(17.960000/2.54, 21.000000/2.54, forward=True)
plt.figure(1).ax_dict[f"{output_folder_figures_calculations}/Figure_2_brainrender_center_crop.png"].set_position([-0.006091, 0.028970, 0.208546, 0.204953])
plt.figure(1).ax_dict[f"{output_folder_figures_calculations}/Figure_2_brainrender_crop.png"].set_position([-0.002836, 0.753232, 0.314292, 0.180476])
plt.figure(1).ax_dict[f"{output_folder_figures_calculations}/Figure_2_brainrender_lateral_crop.png"].set_position([-0.002836, 0.251777, 0.208546, 0.204953])
plt.figure(1).ax_dict[f"{output_folder_figures_calculations}/Figure_2_brainrender_medial_crop.png"].set_position([-0.002836, 0.494937, 0.208546, 0.204953])
plt.figure(1).ax_dict["/alzheimer/Roberto/Allen_Institute/temp/Figure_2_brainrender_center_crop.png"].set_position([-0.006091, 0.016259, 0.208546, 0.204953])
plt.figure(1).ax_dict["/alzheimer/Roberto/Allen_Institute/temp/Figure_2_brainrender_lateral_crop.png"].set_position([-0.002836, 0.255598, 0.208546, 0.204953])
plt.figure(1).ax_dict["/alzheimer/Roberto/Allen_Institute/temp/Figure_2_brainrender_medial_crop.png"].set_position([-0.002836, 0.494937, 0.208546, 0.204953])
plt.figure(1).axes[0].set_position([0.303257, 0.905503, 0.185873, 0.033514])
plt.figure(1).axes[1].set_position([0.303257, 0.863376, 0.185873, 0.033514])
plt.figure(1).axes[2].set_position([0.303257, 0.815772, 0.185873, 0.033514])
plt.figure(1).axes[3].set_position([0.303257, 0.767245, 0.185873, 0.033514])
plt.figure(1).axes[4].set_position([0.303257, 0.727058, 0.185873, 0.033514])
plt.figure(1).axes[5].set_position([0.303257, 0.948404, 0.185873, 0.033514])
plt.figure(1).axes[7].set_position([0.483048, 0.905503, 0.185689, 0.033583])
plt.figure(1).axes[8].set_position([0.483048, 0.862603, 0.185689, 0.033583])
plt.figure(1).axes[9].set_position([0.483048, 0.815276, 0.185689, 0.033583])
plt.figure(1).axes[10].set_position([0.483048, 0.770410, 0.185689, 0.033583])
plt.figure(1).axes[11].set_position([0.483048, 0.725049, 0.185689, 0.033583])
plt.figure(1).axes[12].set_position([0.483048, 0.948404, 0.185689, 0.033583])
plt.figure(1).axes[13].legend(frameon=False)
plt.figure(1).axes[13].set_position([0.728963, 0.755655, 0.257448, 0.220206])
plt.figure(1).axes[13].get_legend()._set_loc((0.015338, 0.701463))
plt.figure(1).axes[14].set_position([0.251434, 0.520373, 0.211754, 0.181121])
plt.figure(1).axes[15].set_position([0.511743, 0.520373, 0.211754, 0.181121])
plt.figure(1).axes[16].legend(frameon=False)
plt.figure(1).axes[16].set_position([0.763513, 0.518769, 0.211754, 0.181121])
plt.figure(1).axes[16].get_legend()._set_loc((0.016887, 0.672921))
plt.figure(1).axes[17].set_position([0.251434, 0.279430, 0.211754, 0.181121])
plt.figure(1).axes[18].set_position([0.511743, 0.279430, 0.211754, 0.181121])
plt.figure(1).axes[19].set_position([0.763513, 0.276222, 0.211754, 0.181121])
plt.figure(1).axes[20].set_position([0.251434, 0.040091, 0.211754, 0.181121])
plt.figure(1).axes[21].set_position([0.512167, 0.040091, 0.211754, 0.181121])
plt.figure(1).axes[22].set_position([0.763513, 0.040091, 0.211754, 0.181121])
plt.figure(1).text(0.5, 0.5, f'Session {session_id}', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_fontsize(6)
plt.figure(1).texts[0].set_ha("center")
plt.figure(1).texts[0].set_position([0.152337, 0.974501])
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_fontname("DejaVu Sans")
plt.figure(1).texts[1].set_position([0.003536, 0.974501])
plt.figure(1).texts[1].set_text("A")
plt.figure(1).texts[1].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.289568, 0.974719])
plt.figure(1).texts[2].set_text("B")
plt.figure(1).texts[2].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_position([0.668317, 0.974501])
plt.figure(1).texts[3].set_text("C")
plt.figure(1).texts[3].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_position([0.003536, 0.711864])
plt.figure(1).texts[4].set_text("D")
plt.figure(1).texts[4].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_position([0.201556, 0.711864])
plt.figure(1).texts[5].set_text("E")
plt.figure(1).texts[5].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_position([0.003536, 0.468523])
plt.figure(1).texts[6].set_text("F")
plt.figure(1).texts[6].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[7].new
plt.figure(1).texts[7].set_position([0.201556, 0.468523])
plt.figure(1).texts[7].set_text("G")
plt.figure(1).texts[7].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[8].new
plt.figure(1).texts[8].set_position([0.003536, 0.233862])
plt.figure(1).texts[8].set_text("H")
plt.figure(1).texts[8].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[9].new
plt.figure(1).texts[9].set_position([0.201556, 0.233862])
plt.figure(1).texts[9].set_text("I")
plt.figure(1).texts[9].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure,  color="#f03a47")  # id=plt.figure(1).texts[10].new
plt.figure(1).texts[10].set_fontsize(6)
plt.figure(1).texts[10].set_position([0.365194, 0.984404])
plt.figure(1).texts[10].set_text("Strong ripple")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure, color="#507dbc")  # id=plt.figure(1).texts[11].new
plt.figure(1).texts[11].set_fontsize(6)
plt.figure(1).texts[11].set_position([0.541946, 0.984404])
plt.figure(1).texts[11].set_text("Common ripple")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure, color=palette_ML["Medial"])  # id=plt.figure(1).texts[12].new
plt.figure(1).texts[12].set_fontsize(7)
plt.figure(1).texts[12].set_ha("center")
plt.figure(1).texts[12].set_position([0.152337, 0.950363])
plt.figure(1).texts[12].set_text("Medial reference")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure, color=palette_ML["Medial"])  # id=plt.figure(1).texts[13].new
plt.figure(1).texts[13].set_fontsize(7)
plt.figure(1).texts[13].set_ha("center")
plt.figure(1).texts[13].set_position([0.105825, 0.711864])
plt.figure(1).texts[13].set_text("Medial reference")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure, color=palette_ML["Central"])  # id=plt.figure(1).texts[14].new
plt.figure(1).texts[14].set_fontsize(7)
plt.figure(1).texts[14].set_ha("center")
plt.figure(1).texts[14].set_position([0.105825, 0.233862])
plt.figure(1).texts[14].set_text("Central reference")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure, color=palette_ML["Lateral"])  # id=plt.figure(1).texts[15].new
plt.figure(1).texts[15].set_fontsize(7)
plt.figure(1).texts[15].set_ha("center")
plt.figure(1).texts[15].set_position([0.105825, 0.468523])
plt.figure(1).texts[15].set_text("Lateral reference")

if Adapt_for_Nature_style is True:
    Naturize()

#% end: automatic generated code from pylustrator
#plt.show()

plt.savefig(f"{output_folder}/Figure_2", dpi=300)