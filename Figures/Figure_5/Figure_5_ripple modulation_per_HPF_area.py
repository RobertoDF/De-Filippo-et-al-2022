from Utils.Settings import minimum_firing_rate_hz, output_folder_figures_calculations, output_folder_calculations,  var_thr
import Utils.Style

import dill
import matplotlib.pyplot as plt


from Utils.Utils import point_plot_modulation_ripples


with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)

fig, axs = plt.subplots(1, 2, figsize=(10,5))

parent_area='HPF'
order= ['CA1',  'CA3', 'DG', 'ProS', 'SUB']
ylim = [-.5,2.5]
dv = 'Ripple modulation (0-50 ms)'
ylabel = 'Early ripple modulation (0-50 ms)'
filter_spiking = (summary_units_df_sub['Firing rate (0-50 ms) medial']>minimum_firing_rate_hz) |\
                                                        (summary_units_df_sub['Firing rate (0-50 ms) lateral']>minimum_firing_rate_hz)
point_plot_modulation_ripples(summary_units_df_sub, dv, parent_area, order, filter_spiking,axs[0], ylabel, ylim )
axs[0].get_legend().remove()
dv = 'Ripple modulation (50-120 ms)'
ylabel = 'Late ripple modulation (50-120 ms)'
point_plot_modulation_ripples(summary_units_df_sub, dv,parent_area, order, filter_spiking, axs[1], ylabel, ylim )



plt.show()