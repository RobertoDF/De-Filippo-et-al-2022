import matplotlib.pyplot as plt
import pylustrator
import dill
from Utils.Style import palette_ML
from Utils.Settings import Adapt_for_Nature_style,root_github_repo, output_folder_supplementary
from Utils.Utils import Naturize

pylustrator.start()

pylustrator.load(f"{root_github_repo}/Figures/Supplementary/Panel_kdeplot_ripple_mod_early_late.py")
pylustrator.load(f"{root_github_repo}/Figures/Supplementary/Panel_kdeplot_ripple_mod_parent_areas.py", offset=[0,1])

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).axes[0].set_position([0.125000, 0.596292, 0.352273, 0.328708])
plt.figure(1).axes[1].set_position([0.547727, 0.596292, 0.352273, 0.328708])
plt.figure(1).axes[2].set_position([0.033575, 0.117344, 0.288099, 0.328708])
plt.figure(1).axes[3].set_position([0.354690, 0.117344, 0.288099, 0.328708])
plt.figure(1).axes[4].set_position([0.675804, 0.117344, 0.288099, 0.328708])
plt.figure(1).text(0.10942492012779552, 0.9496510468594204, 'A', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.011182108626198079, 0.502991026919242, 'B', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[1].new
#% end: automatic generated code from pylustrator
#plt.show()
if Adapt_for_Nature_style is True:
    Naturize()

plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_16", dpi=300)