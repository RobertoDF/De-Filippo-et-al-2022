import dill
import matplotlib.pyplot as plt
from Utils.Settings import output_folder_figures_calculations, output_folder_supplementary, minimum_firing_rate_hz
from Utils.Utils import acronym_color_map, point_plot_modulation_ripples
from Utils.Style import palette_ML
import pylustrator
import pandas as pd

pylustrator.start()

with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)

data = pd.DataFrame(summary_units_df_sub[((summary_units_df_sub['Firing rate (0-50 ms) medial']>minimum_firing_rate_hz) |
                                                        (summary_units_df_sub['Firing rate (0-50 ms) lateral']>minimum_firing_rate_hz))&
                     (summary_units_df_sub['Parent brain region']=='HPF' )].groupby('Brain region')['Ripple type engagement'].value_counts())

fig, axs = plt.subplots(1,5)
axs = axs.ravel()

def my_autopct(pct):
    return ('%.2f'  % pct + " %") if pct > 5 else ''


for q, area in zip(range(5), data.index.get_level_values('Brain region').unique()):

    _, _, autopcts = axs[q].pie(data.loc[area].values.squeeze(), colors=[palette_ML[key] for key in data.loc[area].index], autopct=my_autopct)

    plt.setp(autopcts, **{'color':'white', 'weight':'bold', 'fontsize':3.5})

    axs[q].xaxis.set_label_position('top')
    axs[q].set_title(area, fontsize=3)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).axes[0].texts[3].set_position([0.590624, -0.378913])
plt.figure(1).axes[0].texts[5].set_position([0.686082, -0.096765])
plt.figure(1).axes[1].texts[3].set_position([0.608963, -0.325041])
plt.figure(1).axes[3].texts[3].set_position([0.696500, -0.230858])
plt.figure(1).axes[4].texts[3].set_position([0.597037, -0.259074])
#% end: automatic generated code from pylustrator
plt.show()

plt.savefig(f"{output_folder_supplementary}/Figure 5-Figure supplement 5", dpi=300)