import matplotlib.pyplot as plt
import pylustrator
from Utils.Settings import output_folder_supplementary, output_folder_figures_calculations, Adapt_for_Nature_style
from Utils.Utils import Naturize
import dill

with open(f"{output_folder_figures_calculations}/temp_data_figure_2.pkl", 'rb') as f:
    session_id, session_trajs, columns_to_keep, ripples, real_ripple_summary,\
    lfp_per_probe, ripple_cluster_strong, ripple_cluster_weak, example_session = dill.load(f)


pylustrator.start()

pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Figure_2/Figure_2_summary_medial.py")
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Figure_2/Figure_2_summary_lateral.py")
pylustrator.load("/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Figure_2/Figure_2_summary_center.py")
pylustrator.load(f"{output_folder_figures_calculations}/Figure_2_brainrender_medial_crop.png")
pylustrator.load(f"{output_folder_figures_calculations}/Figure_2_brainrender_lateral_crop.png")
pylustrator.load(f"{output_folder_figures_calculations}/Figure_2_brainrender_center_crop.png")


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(17.940000/2.54, 18.830000/2.54, forward=True)
plt.figure(1).ax_dict["/alzheimer/Roberto/Allen_Institute/temp/Figure_2_brainrender_center_crop.png"].set_position([0.016027, 0.019942, 0.278825, 0.305022])
plt.figure(1).ax_dict["/alzheimer/Roberto/Allen_Institute/temp/Figure_2_brainrender_lateral_crop.png"].set_position([0.016027, 0.350614, 0.278825, 0.305022])
plt.figure(1).ax_dict["/alzheimer/Roberto/Allen_Institute/temp/Figure_2_brainrender_medial_crop.png"].set_position([0.016027, 0.681286, 0.278825, 0.305021])
plt.figure(1).axes[0].set_position([0.346280, 0.694087, 0.178448, 0.292221])
plt.figure(1).axes[1].set_position([0.571816, 0.694087, 0.178448, 0.292221])
plt.figure(1).axes[2].set_position([0.797353, 0.694087, 0.178448, 0.292221])
plt.figure(1).axes[3].set_position([0.348642, 0.363416, 0.177890, 0.292221])
plt.figure(1).axes[4].set_position([0.572374, 0.363416, 0.177890, 0.292221])
plt.figure(1).axes[5].set_position([0.796106, 0.363416, 0.177890, 0.292221])
plt.figure(1).axes[6].set_position([0.348642, 0.032744, 0.178448, 0.292221])
plt.figure(1).axes[7].set_position([0.572374, 0.032744, 0.177890, 0.292221])
plt.figure(1).axes[8].set_position([0.796106, 0.032744, 0.177890, 0.292221])
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.012748, 0.976383])
plt.figure(1).texts[0].set_text("A")
plt.figure(1).texts[0].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.286119, 0.976383])
plt.figure(1).texts[1].set_text("B")
plt.figure(1).texts[1].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.012748, 0.659244])
plt.figure(1).texts[2].set_text("C")
plt.figure(1).texts[2].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_position([0.286119, 0.659244])
plt.figure(1).texts[3].set_text("D")
plt.figure(1).texts[3].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_position([0.012748, 0.324561])
plt.figure(1).texts[4].set_text("E")
plt.figure(1).texts[4].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_position([0.286119, 0.324561])
plt.figure(1).texts[5].set_text("F")
plt.figure(1).texts[5].set_weight("bold")

if Adapt_for_Nature_style is True:
    Naturize()

#% end: automatic generated code from pylustrator
#plt.show()

plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_4", dpi=300)