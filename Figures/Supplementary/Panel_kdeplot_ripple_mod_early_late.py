import dill
import seaborn as sns
import matplotlib.pyplot as plt
from Utils.Settings import output_folder_figures_calculations
from Utils.Utils import  palette_ML, plot_dist_ripple_mod
import pandas as pd
import Utils.Style

with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)

fig, axs = plt.subplots(1, 2, figsize=(10,5))

_ = summary_units_df_sub[['Ripple modulation (0-50 ms) medial', 'Ripple modulation (0-50 ms) lateral', 'Parent brain region']]

_ = pd.wide_to_long(_.reset_index(), stubnames='Ripple modulation (0-50 ms)', i=['Parent brain region','unit_id'], j="Ripple seed", sep=' ', suffix=r'\w+').reset_index()
_['Ripple seed'] = _["Ripple seed"].str.capitalize()

data = _[_['Parent brain region']=='HPF']
param = 'Ripple modulation (0-50 ms)'
plot_dist_ripple_mod(data, param, axs[0])

_ = summary_units_df_sub[['Ripple modulation (50-120 ms) medial', 'Ripple modulation (50-120 ms) lateral', 'Parent brain region']]

_ = pd.wide_to_long(_.reset_index(), stubnames='Ripple modulation (50-120 ms)', i=['Parent brain region','unit_id'], j="Ripple seed", sep=' ', suffix=r'\w+').reset_index()
_['Ripple seed'] = _["Ripple seed"].str.capitalize()

data = _[_['Parent brain region']=='HPF']
param = 'Ripple modulation (50-120 ms)'
plot_dist_ripple_mod(data, param, axs[1])
plt.show()