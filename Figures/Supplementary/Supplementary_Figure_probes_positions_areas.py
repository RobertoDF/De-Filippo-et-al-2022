import dill
import matplotlib.pyplot as plt
import seaborn as sns
from Utils.Settings import output_folder_figures_calculations, output_folder_supplementary, Adapt_for_Nature_style
from Utils.Utils import Naturize
from Utils.Utils import acronym_color_map,  acronym_to_main_area
import Utils.Style

with open(f"{output_folder_figures_calculations}/temp_data_supp_figure_1.pkl", "rb") as fp:  # Unpickling
    data_area, ripple_power, session_id, session_summary, summary_table, ripples = dill.load(fp)

summary_table["Parent area"] = summary_table["Area"].apply(lambda area: acronym_to_main_area(area.split("-")[0]))
summary_table.sort_values(["Graph order"], inplace=True)

palette_areas ={}
for area in summary_table["Parent area"].unique():
    palette_areas[area] = '#' + ''.join(f'{i:02X}' for i in acronym_color_map[area])

ax = sns.PairGrid(data=summary_table, vars=["D-V (µm)", "A-P (µm)", "M-L (µm)"], diag_sharey=False, dropna=True, hue="Parent area", palette=palette_areas)#
ax.map_upper(sns.histplot, alpha=0.5)
ax.map_lower(sns.scatterplot,s=3)
ax.map_diag(sns.kdeplot, alpha=0.5, linewidth=0, fill=True, multiple="stack")
ax = ax.add_legend(fontsize=7)
plt.show()
#plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_1", dpi=300)