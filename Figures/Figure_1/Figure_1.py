import pylustrator
import matplotlib.pyplot as plt
from Utils.Settings import output_folder, output_folder_figures_calculations, Adapt_for_Nature_style
from Utils.Utils import Naturize

pylustrator.start()

pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Figure_1/Figure_1_relplot_corrs.py", offset=[0, 0])
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Figure_1/Figure_1_corr_distance_power_CA1_CA1.py", offset=[0, 0.2])
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Figure_1/Figure_1_violinplot_quartiles.py", offset=[0, 0.4])
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Figure_1/Figure_1_distribution_CA1_CA1_corr.py", offset=[0, 0.2])
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Figure_1/Figure_1_pointplot_lag.py", offset=[0, 0.2])
pylustrator.load(f"{output_folder_figures_calculations}/Figure_1_brainrender_high_dist.png")
pylustrator.load(f"{output_folder_figures_calculations}/Figure_1_brainrender_low_dist.png")
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Figure_1/Figure_1_scatter_distribution.py", offset=[0, 0.2])
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Figure_1/Figure_1_pointplot_lag_summary.py", offset=[0, 0.2])

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(18.000000/2.54, 18.000000/2.54, forward=True)
plt.figure(1).ax_dict[f"{output_folder_figures_calculations}/Figure_1_brainrender_high_dist.png"].set_position([0.413958, 0.528845, 0.265743, 0.171437])
plt.figure(1).ax_dict[f"{output_folder_figures_calculations}/Figure_1_brainrender_low_dist.png"].set_position([0.413958, 0.361373, 0.265743, 0.171437])
plt.figure(1).ax_dict["/alzheimer/Roberto/Allen_Institute/temp/Figure_1_brainrender_high_dist.png"].set_position([0.371515, 0.528845, 0.249540, 0.171437])
plt.figure(1).ax_dict["/alzheimer/Roberto/Allen_Institute/temp/Figure_1_brainrender_low_dist.png"].set_position([0.371515, 0.361373, 0.249540, 0.171437])
plt.figure(1).axes[0].set_position([0.137083, 0.859636, 0.111215, 0.111215])
plt.figure(1).axes[1].set_position([0.271798, 0.859636, 0.111215, 0.111215])
plt.figure(1).axes[2].set_position([0.135671, 0.720685, 0.111215, 0.111215])
plt.figure(1).axes[3].set_position([0.270385, 0.720685, 0.111215, 0.111215])
plt.figure(1).axes[4].set_position([0.451710, 0.737634, 0.227991, 0.227991])
plt.figure(1).axes[5].set_position([0.081453, 0.396235, 0.263091, 0.240610])
plt.figure(1).axes[6].set_position([0.748398, 0.742860, 0.227991, 0.227991])
plt.figure(1).axes[7].set_position([0.760896, 0.228540, 0.204136, 0.106036])
plt.figure(1).axes[8].set_position([0.451710, 0.228366, 0.204136, 0.106036])
plt.figure(1).axes[9].set_position([0.760953, 0.060323, 0.204079, 0.106210])
plt.figure(1).axes[10].set_position([0.451767, 0.060323, 0.204079, 0.106210])
plt.figure(1).axes[13].set_position([0.683269, 0.615844, 0.293120, 0.084438])
plt.figure(1).axes[14].set_position([0.683269, 0.506213, 0.293120, 0.084438])
plt.figure(1).axes[15].set_position([0.683269, 0.396583, 0.293120, 0.084438])
plt.figure(1).axes[15].get_legend()._set_loc((0.495298, 0.089560))
plt.figure(1).axes[16].set_position([0.126914, 0.228366, 0.204079, 0.106210])
plt.figure(1).axes[17].set_position([0.126914, 0.060497, 0.204079, 0.106210])
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.004237, 0.973164])
plt.figure(1).texts[0].set_text("A")
plt.figure(1).texts[0].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.406780, 0.973164])
plt.figure(1).texts[1].set_text("B")
plt.figure(1).texts[1].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.696761, 0.973130])
plt.figure(1).texts[2].set_text("C")
plt.figure(1).texts[2].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_position([0.004237, 0.658192])
plt.figure(1).texts[3].set_text("D")
plt.figure(1).texts[3].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_position([0.343220, 0.658192])
plt.figure(1).texts[4].set_text("E")
plt.figure(1).texts[4].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_position([0.618644, 0.675141])
plt.figure(1).texts[5].set_text("F")
plt.figure(1).texts[5].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_position([0.004237, 0.341808])
plt.figure(1).texts[6].set_text("G")
plt.figure(1).texts[6].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[7].new
plt.figure(1).texts[7].set_position([0.343220, 0.341808])
plt.figure(1).texts[7].set_text("H")
plt.figure(1).texts[7].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[8].new
plt.figure(1).texts[8].set_position([0.696761, 0.341808])
plt.figure(1).texts[8].set_text("I")
plt.figure(1).texts[8].set_weight("bold")

if Adapt_for_Nature_style is True:
    Naturize()

#% end: automatic generated code from pylustrator
#plt.show()
plt.savefig(f"{output_folder}/Figure_1", dpi=300)

