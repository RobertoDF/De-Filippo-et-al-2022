import matplotlib.pyplot as plt
import pylustrator
from Utils.Settings import root_github_repo,output_folder_supplementary, output_folder_figures_calculations, root_github_repo, Adapt_for_Nature_style
from Utils.Utils import Naturize

# to recompute data run both:Util_Supplementary_brainrender_limits.py & Util_Supplementary_Figure_ML_sections_limits.py

pylustrator.start()

pylustrator.load(f'{ output_folder_figures_calculations}/brainrender_hippocampal_sectors_crop.png', offset=[0, 0])
pylustrator.load(f'{root_github_repo}/Figures/Supplementary/Supplementary_Figure_table_distance_variance_explained.py')
pylustrator.load(f'{root_github_repo}/Figures/Supplementary/Panel_histogram_LR.py')

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(8.880000/2.54, 10.000000/2.54, forward=True)
plt.figure(1).ax_dict["/alzheimer/Roberto/Allen_Institute/temp/brainrender_hippocampal_sectors_crop.png"].set_position([-0.005905, 0.006345, 0.998852, 0.572062])
plt.figure(1).axes[1].set_position([0.134246, 0.711503, 0.779388, 0.402074])
plt.figure(1).axes[2].set_position([0.112364, 0.624464, 0.801270, 0.190343])
plt.figure(1).axes[2].texts[2].set_position([0.400000, 0.900000])
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.004052, 0.967626])
plt.figure(1).texts[0].set_text("A")
plt.figure(1).texts[0].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.004052, 0.795773])
plt.figure(1).texts[1].set_text("B")
plt.figure(1).texts[1].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.004052, 0.527444])
plt.figure(1).texts[2].set_text("C")
plt.figure(1).texts[2].set_weight("bold")
#% end: automatic generated code from pylustrator
#plt.show()

plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_5", dpi=300)