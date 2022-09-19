import matplotlib.pyplot as plt
import pylustrator
from Utils.Style import palette_ML, palette_timelags
from Utils.Settings import output_folder, output_folder_figures_calculations, Adapt_for_Nature_style
from Utils.Utils import Naturize

pylustrator.start()

pylustrator.load("Figure_3_seed_by_ML_strong.py", offset=[0, 0])
pylustrator.load("Figure_3_seed_by_ML_common.py", offset=[0, 0])
pylustrator.load("Figure_3_common_strong_comparison_medial.py", offset=[0, 0])
pylustrator.load("Figure_3_common_strong_comparison_central.py", offset=[0, 0])
pylustrator.load("Figure_3_common_strong_comparison_lateral.py", offset=[0, 0])


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(18.30000/2.54, 10.980000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.532112, 0.515109, 0.437445, 0.379419])
plt.figure(1).axes[0].get_legend()._set_loc((0.741724, 0.855067))
plt.figure(1).axes[1].set_position([0.043723, 0.515109, 0.437445, 0.379419])
plt.figure(1).axes[1].get_legend()._set_loc((0.708553, 0.729687))
plt.figure(1).axes[2].set_position([0.043723, 0.065229, 0.273403, 0.364891])
plt.figure(1).axes[3].set_position([0.369938, 0.065229, 0.273403, 0.364891])
plt.figure(1).axes[3].get_legend()._set_loc((0.239827, -0.004341))
plt.figure(1).axes[4].set_position([0.696154, 0.065229, 0.273403, 0.364891])
plt.figure(1).axes[4].get_legend()._set_loc((0.318790, 0.026458))
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.002903, 0.969734])
plt.figure(1).texts[0].set_text("A")
plt.figure(1).texts[0].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.002903, 0.462470])
plt.figure(1).texts[1].set_text("B")
plt.figure(1).texts[1].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure, color=palette_timelags["Common ripples"])  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_fontsize(7)
plt.figure(1).texts[2].set_ha("center")
plt.figure(1).texts[2].set_position([0.262445, 0.859329])
plt.figure(1).texts[2].set_text("Common ripples")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure, color=palette_timelags["Strong ripples"])  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_fontsize(8)
plt.figure(1).texts[3].set_ha("center")
plt.figure(1).texts[3].set_position([0.750835, 0.859329])
plt.figure(1).texts[3].set_text("Strong ripples")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure, color=palette_ML["Medial"])  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_fontsize(8)
plt.figure(1).texts[4].set_ha("center")
plt.figure(1).texts[4].set_position([0.180425, 0.438257])
plt.figure(1).texts[4].set_text("Medial reference")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure, color=palette_ML["Central"])  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_fontsize(8)
plt.figure(1).texts[5].set_ha("center")
plt.figure(1).texts[5].set_position([0.506640, 0.438257])
plt.figure(1).texts[5].set_text("Central reference")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure, color=palette_ML["Lateral"])  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_fontsize(8)
plt.figure(1).texts[6].set_ha("center")
plt.figure(1).texts[6].set_position([0.832856, 0.438257])
plt.figure(1).texts[6].set_text("Lateral reference")

if Adapt_for_Nature_style is True:
    Naturize()

#% end: automatic generated code from pylustrator
#plt.show()
plt.savefig(f"{output_folder}/Figure_3", dpi=300)