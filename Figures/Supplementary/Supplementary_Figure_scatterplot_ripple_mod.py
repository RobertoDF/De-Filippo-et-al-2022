import matplotlib.pyplot as plt
import pylustrator
import dill
from Utils.Style import palette_ML
from Utils.Settings import Adapt_for_Nature_style,root_github_repo, output_folder_supplementary
from Utils.Utils import Naturize

pylustrator.start()

pylustrator.load(f"{root_github_repo}/Figures/Supplementary/Panel_scatterplot_ripple_mod_parent_areas_medial.py")
pylustrator.load(f"{root_github_repo}/Figures/Supplementary/Panel_scatterplot_ripple_mod_parent_areas_lateral.py", offset=[0,1])

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).text(0.005503144654088052, 0.9677165354330709, 'A', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.005503144654088052, 0.4968503937007874, 'B', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[1].new
plt.figure(1).text(0.4677672955974843, 0.9677165354330709, 'Medial ripple seed', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).text(0.4677672955974843, 0.4968503937007874, 'Lateral ripple seed', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
#% end: automatic generated code from pylustrator
#plt.show()

plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_14", dpi=300)

