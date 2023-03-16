from Utils.Settings import output_folder_figures_calculations, output_folder_calculations,  var_thr
import Utils.Style
import pandas as pd
import dill
import matplotlib.pyplot as plt
from Utils.Utils import point_plot_modulation_ripples

with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)



#axs[1].text(.6, .7, 'Cohen's d: ' + str(round(ttest_late_spiking['cohen-d'].values[0], 2)), transform=axs[1].transAxes,fontsize=6, ha='center', va='center');

fig, axs = plt.subplots(1, figsize=(5,5))

parent_area='HPF'
order= ['CA1',  'CA3', 'DG', 'ProS', 'SUB']
dv = 'Pre-ripple modulation'
ylabel='Pre-ripple modulation (20-0ms)'
filter_spiking = summary_units_df_sub['Firing rate (20-0 ms) medial']>0.025
ylim = [-.25, .5]
point_plot_modulation_ripples(summary_units_df_sub, dv,parent_area,order,filter_spiking, axs, ylabel, ylim)

plt.show()