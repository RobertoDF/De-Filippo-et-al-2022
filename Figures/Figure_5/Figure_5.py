import matplotlib.pyplot as plt
import pylustrator
import dill
from Utils.Style import palette_ML
from Utils.Settings import output_folder, output_folder_figures_calculations, Adapt_for_Nature_style,root_github_repo, output_folder_figures_calculations
from Utils.Utils import Naturize

pylustrator.start()

pylustrator.load(f"{root_github_repo}/Figures/Figure_5/Figure_5_kde_HPF_ripple_modulated.py")
pylustrator.load(f"{root_github_repo}/Figures/Figure_5/Figure_5_scatter_ripple_modulated.py", offset=[0,1])
pylustrator.load(f"{root_github_repo}/Figures/Figure_5/Figure_5_neurons_distribution_ML.py", offset=[0,1])
pylustrator.load(f"{output_folder_figures_calculations}/Figure_5_brainrender_cells.png")
pylustrator.load(f"{root_github_repo}/Figures/Figure_5/Figure_5_ripple modulation_per_HPF_area.py")
pylustrator.load(f"{root_github_repo}/Figures/Figure_5/Figure_5_preripple modulation_per_HPF_area.py")
pylustrator.load(f"{root_github_repo}/Figures/Figure_5/Figure_5_ripple_specific_modulation_scatterplot.py")
pylustrator.load(f"{root_github_repo}/Figures/Figure_5/Figure_5_ripple_specific_modulation_pie.py", offset=[.5, .5])

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(18/2.54, 18/2.54, forward=True)
plt.figure(1).ax_dict["/alzheimer/Roberto/Allen_Institute/temp/Figure_5_brainrender_cells.png"].set_position([0.662764, 0.768361, 0.322524, 0.215934])
plt.figure(1).axes[0].set_position([0.365154, 0.733812, 0.273050, 0.250483])
plt.figure(1).axes[1].set_position([0.045381, 0.733812, 0.278315, 0.250483])
plt.figure(1).axes[2].set_position([0.683038, 0.592709, 0.302250, 0.166989])
plt.figure(1).axes[3].set_position([0.683038, 0.396432, 0.302250, 0.166989])
plt.figure(1).axes[3].get_legend()._set_loc((0.758021, 1.487929))
plt.figure(1).axes[5].set_position([0.045381, 0.396432, 0.278315, 0.250483])
plt.figure(1).axes[6].set_position([0.368530, 0.396432, 0.269674, 0.250483])
plt.figure(1).axes[7].set_ylim(0.0, 3.0)
plt.figure(1).axes[7].set_position([0.045381, 0.059053, 0.278315, 0.250483])
plt.figure(1).axes[8].set_position([0.410592, 0.059053, 0.269674, 0.250483])
plt.figure(1).axes[8].get_legend()._set_loc((0.364736, 0.762754))
plt.figure(1).axes[9].set_position([0.683038, -0.000317, 0.327780, 0.327780])
plt.figure(1).axes[9].title.set_position([0.500000, 0.928975])
plt.figure(1).axes[9].texts[1].set_position([-0.263583, 0.219591])
plt.figure(1).text(0.48758235782825293, 0.9648019132785688, 'HPF', transform=plt.figure(1).transFigure, color='#7ed04b', weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.0049504950495049905, 0.980905233380481, 'A', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[1].new
plt.figure(1).text(0.34582743988684583, 0.980905233380481, 'B', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[2].new
plt.figure(1).text(0.6584158415841584, 0.980905233380481, 'C', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[3].new
plt.figure(1).text(0.0049504950495049506, 0.6499292786421497, 'D', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[4].new
plt.figure(1).text(0.0049504950495049506, 0.32602545968882596, 'E', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[5].new
plt.figure(1).text(0.3458274398868458, 0.32602545968882596, 'F', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[6].new
#% end: automatic generated code from pylustrator
#plt.show()


if Adapt_for_Nature_style is True:
    Naturize()
plt.savefig(f"{output_folder}/Figure_5", dpi=300)