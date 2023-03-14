import dill
import seaborn as sns
import matplotlib.pyplot as plt
from Utils.Settings import output_folder_figures_calculations
from Utils.Utils import acronym_color_map
import Utils.Style

with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)

summary_units_df_sub["Firing rate (0-50 ms)"] = (summary_units_df_sub["Firing rate (0-50 ms) medial"]+summary_units_df_sub["Firing rate (0-50 ms) lateral"])/2
summary_units_df_sub["Firing rate (100-0 ms)"] = (summary_units_df_sub["Firing rate (100-0 ms) medial"]+summary_units_df_sub["Firing rate (100-0 ms) lateral"])/2

palette_parent_areas = dict()
for area in summary_units_df_sub["Parent brain region"].unique():
    palette_parent_areas[area] = '#' + ''.join(f'{i:02X}' for i in acronym_color_map[area]);


fig, axs = plt.subplots(1, figsize=(5,5))
sns.scatterplot(ax=axs, data=summary_units_df_sub, x='Firing rate (0-50 ms)', y='Firing rate (100-0 ms)', hue='Parent brain region', palette=palette_parent_areas, alpha=.5, s=5,
                hue_order=['HPF', 'Isocortex', 'MB', 'TH'])
axs.plot([0, axs.get_ylim()[1]], [0, axs.get_ylim()[1]],  alpha=.5, linestyle='--', color='k')
plt.show()