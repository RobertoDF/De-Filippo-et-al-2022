import matplotlib.pyplot as plt
import pylustrator
from Utils.Settings import output_folder_supplementary, Adapt_for_Nature_style
from Utils.Utils import Naturize
import Utils.Style

#pylustrator.start()

pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Panel_summary_heatmap_spike_differences_interp_exc.py")
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Panel_summary_heatmap_spike_differences_interp_inh.py", offset=[1, 0])
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Panel_summary_heatmap_spike_differences_interp_medial.py", offset=[0, 1])
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Panel_summary_heatmap_spike_differences_interp_lateral.py", offset=[.5, .5])
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Panel_summary_heatmap_spike_differences_interp_common.py", offset=[0, 1])
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Panel_summary_heatmap_spike_differences_interp_strong.py", offset=[.5, .66])

if Adapt_for_Nature_style is True:
    Naturize()

plt.figure(1).set_size_inches(17.940000/2.54, 18.830000/2.54, forward=True)
#plt.show()

plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_10", dpi=300)
