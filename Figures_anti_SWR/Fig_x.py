import pylustrator
import matplotlib.pyplot as plt
from Utils.Settings import output_folder, output_folder_figures_calculations, Adapt_for_Nature_style
from Utils.Utils import Naturize

pylustrator.start()

pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures_anti_SWR/anti_SWR_probability.py", offset=[0, 0])
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures_anti_SWR/anti_SWR_probability_HPF.py", offset=[0, 0.2])
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures_anti_SWR/stacked_SWR_modulation.py", offset=[0, 0.4])
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures_anti_SWR/SWR_modulation_unstacked.py", offset=[0, 0.2])

pylustrator.load(f"/home/roberto/Github/De-Filippo-et-al-2022/Figures_anti_SWR/anti_SWR_waveform_dur.py")

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(18.000000/2.54, 12.000000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.507833, 0.670850, 0.206078, 0.301371])
plt.figure(1).axes[1].set_position([0.781575, 0.731234, 0.203136, 0.240987])
plt.figure(1).axes[2].set_position([0.781575, 0.432026, 0.203136, 0.240987])
plt.figure(1).axes[3].set_xlim(-1.0, 2.0)
plt.figure(1).axes[3].set_position([0.051621, 0.670850, 0.395247, 0.301371])
plt.figure(1).axes[4].set_position([0.051621, 0.432026, 0.139019, 0.151031])
plt.figure(1).axes[5].set_position([0.229332, 0.432026, 0.139019, 0.151031])
plt.figure(1).axes[6].set_position([0.407044, 0.432026, 0.139019, 0.151031])
plt.figure(1).axes[7].set_position([0.584755, 0.432026, 0.139019, 0.151031])
plt.figure(1).axes[8].set_position([0.051621, 0.061542, 0.194626, 0.301222])
plt.figure(1).axes[9].set_position([0.307044, 0.061542, 0.194626, 0.301222])
plt.figure(1).axes[10].set_position([0.562466, 0.061542, 0.194626, 0.301222])
plt.figure(1).axes[11].set_position([0.817889, 0.061542, 0.166822, 0.301222])

#% end: automatic generated code from pylustrator

plt.figure(1).text(0.005503144654088059, 0.9709371293001192, 'A', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.005503144654088059, 0.9709371293001192, 'B', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.005503144654088059, 0.9709371293001192, 'C', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.005503144654088059, 0.9709371293001192, 'D', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.005503144654088059, 0.9709371293001192, 'E', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.005503144654088059, 0.9709371293001192, 'F', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.005503144654088059, 0.9709371293001192, 'G', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.005503144654088059, 0.9709371293001192, 'H', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[1].set_position([0.451258, 0.970937])
plt.figure(1).texts[2].set_position([0.731918, 0.970937])
plt.figure(1).texts[3].set_position([0.005503, 0.597272])
plt.figure(1).texts[4].set_position([0.005503, 0.349244])
plt.figure(1).texts[5].set_position([0.261006, 0.349244])
plt.figure(1).texts[6].set_position([0.505503, 0.349244])
plt.figure(1).texts[7].set_position([0.759889, 0.349244])

#plt.show()
plt.savefig(f"{output_folder}/Figure_x", dpi=300)