import matplotlib.pyplot as plt
import pylustrator
from Utils.Settings import output_folder_supplementary, output_folder, Adapt_for_Nature_style
from Utils.Utils import Naturize
import Utils.Style

pylustrator.start()

pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Panel_clusters_per_ripple_early_late_by_neuron_type.py")
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Panel_spiking_rate_early_late_by_neuron_type.py")

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(8.900000/2.54, 8.900000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.125000, 0.113318, 0.337458, 0.345966])
plt.figure(1).axes[1].set_position([0.547727, 0.113318, 0.337458, 0.345966])
plt.figure(1).axes[1].get_legend()._set_loc((0.564268, 0.987910))
plt.figure(1).axes[2].set_ylim(0.043275815698154654, 0.5121126146247941)
plt.figure(1).axes[2].set_yticks([0.1, 0.2, 0.30000000000000004, 0.4, 0.5])
plt.figure(1).axes[2].set_yticklabels(["0.1", "0.2", "0.3", "0.4", "0.5"], fontsize=5)
plt.figure(1).axes[2].set_position([0.125000, 0.589969, 0.337458, 0.345966])
plt.figure(1).axes[2].yaxis.labelpad = -0.721239
plt.figure(1).axes[2].yaxis.labelpad = 2.0
plt.figure(1).axes[3].set_position([0.546621, 0.589969, 0.337458, 0.345966])
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.020000, 0.954469])
plt.figure(1).texts[0].set_text("A")
plt.figure(1).texts[0].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.020000, 0.470777])
plt.figure(1).texts[1].set_text("B")
plt.figure(1).texts[1].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_fontsize(6)
plt.figure(1).texts[2].set_ha("center")
plt.figure(1).texts[2].set_position([0.293729, 0.973451])
plt.figure(1).texts[2].set_text("Early phase (0-50 ms)")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_fontsize(6)
plt.figure(1).texts[3].set_ha("center")
plt.figure(1).texts[3].set_position([0.707635, 0.973451])
plt.figure(1).texts[3].set_text("Late phase (50-120 ms)")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_fontsize(6)
plt.figure(1).texts[4].set_position([0.316372, 0.442478])
plt.figure(1).texts[4].set_text("*")
plt.figure(1).texts[4].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_fontsize(6)
plt.figure(1).texts[5].set_position([0.740044, 0.442478])
plt.figure(1).texts[5].set_text("*")
plt.figure(1).texts[5].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_fontsize(6)
plt.figure(1).texts[6].set_position([0.316372, 0.918142])
plt.figure(1).texts[6].set_text("*")
plt.figure(1).texts[6].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[7].new
plt.figure(1).texts[7].set_fontsize(6)
plt.figure(1).texts[7].set_position([0.740044, 0.918142])
plt.figure(1).texts[7].set_text("*")
plt.figure(1).texts[7].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[8].new
plt.figure(1).texts[8].set_fontsize(6)
plt.figure(1).texts[8].set_position([0.216817, 0.420354])
plt.figure(1).texts[8].set_text("*")
plt.figure(1).texts[8].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[9].new
plt.figure(1).texts[9].set_fontsize(6)
plt.figure(1).texts[9].set_position([0.639385, 0.422089])
plt.figure(1).texts[9].set_text("*")
plt.figure(1).texts[9].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[10].new
plt.figure(1).texts[10].set_fontsize(6)
plt.figure(1).texts[10].set_position([0.779868, 0.404289])
plt.figure(1).texts[10].set_text("*")
plt.figure(1).texts[10].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[11].new
plt.figure(1).texts[11].set_fontsize(6)
plt.figure(1).texts[11].set_position([0.256637, 0.881637])
plt.figure(1).texts[11].set_text("*")
plt.figure(1).texts[11].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[12].new
plt.figure(1).texts[12].set_fontsize(6)
plt.figure(1).texts[12].set_position([0.778761, 0.881637])
plt.figure(1).texts[12].set_text("*")
plt.figure(1).texts[12].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[13].new
plt.figure(1).texts[13].set_fontsize(6)
plt.figure(1).texts[13].set_position([0.639385, 0.898740])
plt.figure(1).texts[13].set_text("*")
plt.figure(1).texts[13].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[14].new
plt.figure(1).texts[14].set_fontsize(6)
plt.figure(1).texts[14].set_position([0.639385, 0.239655])
plt.figure(1).texts[14].set_text("*")
plt.figure(1).texts[14].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[15].new
plt.figure(1).texts[15].set_fontsize(6)
plt.figure(1).texts[15].set_position([0.639385, 0.685011])
plt.figure(1).texts[15].set_text("*")
plt.figure(1).texts[15].set_weight("bold")
#% end: automatic generated code from pylustrator
#plt.show()

plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_10", dpi=300)


