import dill
import seaborn as sns
import matplotlib.pyplot as plt
from Utils.Settings import output_folder_figures_calculations
from Utils.Utils import acronym_color_map
import Utils.Style
import scipy as sp

with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)

summary_units_df_sub["Firing rate (0-50 ms)"] = (summary_units_df_sub["Firing rate (0-50 ms) medial"]+summary_units_df_sub["Firing rate (0-50 ms) lateral"])/2
summary_units_df_sub["Firing rate (120-0 ms)"] = (summary_units_df_sub["Firing rate (120-0 ms) medial"]+summary_units_df_sub["Firing rate (120-0 ms) lateral"])/2

palette_parent_areas = dict()
for area in summary_units_df_sub["Parent brain region"].unique():
    palette_parent_areas[area] = '#' + ''.join(f'{i:02X}' for i in acronym_color_map[area]);


g = sns.lmplot(data=summary_units_df_sub, x='Firing rate (0-50 ms)', y='Firing rate (120-0 ms)',
               hue='Parent brain region',
               palette=palette_parent_areas, scatter_kws=dict(alpha=.3, s=3),
               hue_order=['HPF', 'Isocortex', 'MB', 'TH'], height=5, aspect=1, legend=False)

yg = {'HPF': .8, 'Isocortex': .85, 'MB': .9, 'TH': .95}


def annotate(data, **kws):
    g = data['Parent brain region'].unique()[0]

    # get the y-position from the dict
    y = yg[g]

    r, p = sp.stats.pearsonr(data['Firing rate (0-50 ms)'], data['Firing rate (120-0 ms)'])
    ax = plt.gca()
    if p == 0:
        ax.text(.03, y, f'{g}: R²={r ** 2:.2f}, p<0.0005', fontsize=5,  transform=ax.transAxes, color=palette_parent_areas[g])
    else:
        ax.text(.03, y, f'{g}: R²={r ** 2:.2f}, p={2:.2e}', fontsize=5, transform=ax.transAxes, color=palette_parent_areas[g])


g.map_dataframe(annotate)
g.map_dataframe(lambda data, **kws: plt.axline((0, 0), slope=1,  alpha=.5, linestyle='--', color='k'))
plt.show()