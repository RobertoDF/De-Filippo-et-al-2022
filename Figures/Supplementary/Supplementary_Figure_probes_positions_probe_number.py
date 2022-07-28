import dill
import matplotlib.pyplot as plt
import seaborn as sns
from Utils.Settings import output_folder_figures_calculations, output_folder_supplementary
from Utils.Utils import acronym_structure_path_map, summary_structures, summary_structures_finer
import Utils.Style

my_colors = ["#301A4B", "#087E8B", "#F59A8C", "#5FAD56", "#AFA2FF", "#FAC748"]
# Set your custom color palette
my_palette = sns.color_palette(my_colors)

with open(f"{output_folder_figures_calculations}/temp_data_supp_figure_1.pkl", "rb") as fp:  # Unpickling
    data_area, ripple_power, session_id, session_summary, summary_table, ripples = dill.load(fp)

ax = sns.PairGrid(data=summary_table, vars=["D-V (µm)", "A-P (µm)", "M-L (µm)"], diag_sharey=False, dropna=True, hue="Probe number", palette="Dark2")#
ax.map_upper(sns.kdeplot, alpha=0.5)
ax.map_lower(sns.scatterplot,s=3)
ax.map_diag(sns.kdeplot, alpha=0.5, linewidth=0, fill=True, multiple="stack")
ax = ax.add_legend(fontsize=7)
plt.show()
plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_probe_positions_probe_number", dpi=300)