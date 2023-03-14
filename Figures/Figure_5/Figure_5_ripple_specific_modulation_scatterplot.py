from Utils.Settings import output_folder_figures_calculations
import Utils.Style
from Utils.Style import palette_ML
import pandas as pd
import dill
import matplotlib.pyplot as plt
from Utils.Utils import point_plot_modulation_ripples
import seaborn as sns

with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)

palette_ML['Lateral ripple engagement'] = (0.29408557, 0.13721193, 0.38442775)
palette_ML['Medial ripple engagement'] = (0.92891402, 0.68494686, 0.50173994)
palette_ML['No preference'] = (0.8,.8,.8)


fig, axs = plt.subplots(1,  figsize=(5,5))
sns.scatterplot(data=summary_units_df_sub[(summary_units_df_sub['Firing rate (0-120 ms) medial']>0.025)&
                                          (summary_units_df_sub['Parent brain region']=='HPF' )],
               x='Ripple modulation (0-50 ms) medial', y= 'Ripple modulation (0-50 ms) lateral', alpha=.5, ax=axs, s=8, hue='Ripple type engagement', palette=palette_ML)
axs.plot([0, axs.get_ylim()[1]], [0, axs.get_ylim()[1]],  alpha=.5, linestyle='--', color='k')
axs.plot([0, axs.get_ylim()[1]], [0, axs.get_ylim()[1]/2],  alpha=.5, linestyle='--', color='k')
axs.plot([0, axs.get_ylim()[1]/2], [0, axs.get_ylim()[1]],  alpha=.5, linestyle='--', color='k')
axs.set_xlim((0,axs.get_xlim()[1]))
axs.set_ylim((0,axs.get_ylim()[1]))

plt.show()