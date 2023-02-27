import matplotlib.pyplot as plt
import pylustrator
from Utils.Settings import output_folder_supplementary, Adapt_for_Nature_style
from Utils.Utils import Naturize
import Utils.Style

pylustrator.start()

pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Panel_summary_heatmap_spike_differences_interp_inh.py")
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Panel_summary_heatmap_spike_differences_interp_exc.py")

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(8.900000/2.54, 8.900000/2.54, forward=True)
plt.figure(1).ax_dict["colorbar_exc"].set_position([0.813469, 0.640263, 0.011261, 0.225221])
plt.figure(1).ax_dict["colorbar_inh"].set_position([0.813469, 0.135280, 0.011261, 0.225221])
plt.figure(1).axes[0].set_position([0.264535, 0.078992, 0.450442, 0.337832])
plt.figure(1).axes[2].set_position([0.264535, 0.585083, 0.450442, 0.337832])
plt.figure(1).axes[2].title.set_fontsize(7)
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure, color="#D64933")  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_fontsize(7)
plt.figure(1).texts[0].set_position([0.081395, 0.624031])
plt.figure(1).texts[0].set_rotation(90.0)
plt.figure(1).texts[0].set_text("Putative excitatory")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure, color="#00C2D1")  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_fontsize(7)
plt.figure(1).texts[1].set_position([0.081395, 0.129014])
plt.figure(1).texts[1].set_rotation(90.0)
plt.figure(1).texts[1].set_text("Putative inhibitory")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.081395, 0.945561])
plt.figure(1).texts[2].set_text("A")
plt.figure(1).texts[2].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_position([0.081395, 0.439698])
plt.figure(1).texts[3].set_text("B")
plt.figure(1).texts[3].set_weight("bold")

if Adapt_for_Nature_style is True:
    Naturize()

#% end: automatic generated code from pylustrator
plt.show()

#plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_10", dpi=300)
