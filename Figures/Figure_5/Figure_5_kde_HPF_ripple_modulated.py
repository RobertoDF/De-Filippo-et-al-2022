import dill
import seaborn as sns
import matplotlib.pyplot as plt
from Utils.Settings import output_folder_figures_calculations
from Utils.Utils import acronym_color_map, palette_ML
import pandas as pd
import Utils.Style

with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)

_ = summary_units_df_sub[['Ripple modulation (0-50 ms) medial', 'Ripple modulation (0-50 ms) lateral', 'Parent brain region']]

_ = pd.wide_to_long(_.reset_index(), stubnames='Ripple modulation (0-50 ms)', i=['Parent brain region','unit_id'], j='Type', sep=' ', suffix=r'\w+').reset_index()
_['Type'] = _['Type'].str.capitalize()

fig, axs = plt.subplots(1, figsize=(5,5))
g = sns.kdeplot(data=_[_['Parent brain region']=='HPF'], x='Ripple modulation (0-50 ms)', hue='Type', palette=palette_ML, ax=axs, fill=True, gridsize=500, cut=0)
axs.set_xlim((0, 15))
axs.axvline(1,color= 'k', linestyle='--')
axs.axvline(2,color= 'r', linestyle='--')
axs.get_yaxis().set_visible(False)

axs.spines[['left']].set_visible(False)
plt.show()