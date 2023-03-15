import dill
import matplotlib.pyplot as plt
from Utils.Settings import output_folder_figures_calculations, output_folder_supplementary
from Utils.Utils import acronym_color_map, point_plot_modulation_ripples
from Utils.Style import palette_ML
import pylustrator
import pandas as pd

pylustrator.start()

with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)

data = pd.DataFrame(summary_units_df_sub[(summary_units_df_sub['Firing rate (0-120 ms) medial']>0.025)&
                     (summary_units_df_sub['Parent brain region']=='HPF' )].groupby('Brain region')['Ripple type engagement'].value_counts())

fig, axs = plt.subplots(1,5)
axs = axs.ravel()

def my_autopct(pct):
    return ('%.2f'  % pct + " %") if pct > 20 else ''


for q, area in zip(range(5), data.index.get_level_values('Brain region').unique()):

    _, _, autopcts = axs[q].pie(data.loc[area].values.squeeze(), colors=[palette_ML[key] for key in data.loc[area].index], autopct=my_autopct)

    plt.setp(autopcts, **{'color':'white', 'weight':'bold', 'fontsize':3.5})

    axs[q].xaxis.set_label_position('top')
    axs[q].set_title('Ripple type engagement', fontsize=3)

#plt.show()

plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_18", dpi=300)