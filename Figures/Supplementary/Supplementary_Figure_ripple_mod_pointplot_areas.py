import dill
import matplotlib.pyplot as plt
from Utils.Settings import output_folder_figures_calculations, output_folder_supplementary, minimum_firing_rate_hz
from Utils.Utils import acronym_color_map, point_plot_modulation_ripples
import Utils.Style
import pylustrator

pylustrator.start()

with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)

palette_areas = dict()
for area in summary_units_df_sub['Brain region'].unique():
    palette_areas[area] = '#' + ''.join(f'{i:02X}' for i in acronym_color_map[area]);
fig, axs = plt.subplots(3, 2, figsize=(10,15))


parent_area='Isocortex'
order= ['VIS', 'VISam', 'VISpm', 'VISp', 'VISl', 'VISrl', 'VISal']
ylim= [-0.25,0.5]
dv = 'Ripple modulation (0-50 ms)'
ylabel = 'Early ripple modulation (0-50 ms)'
filter_spiking = (summary_units_df_sub['Firing rate (0-50 ms) medial']>minimum_firing_rate_hz) |\
                                                        (summary_units_df_sub['Firing rate (0-50 ms) lateral']>minimum_firing_rate_hz)
point_plot_modulation_ripples(summary_units_df_sub, dv, parent_area, order,filter_spiking,axs[0,0], ylabel,ylim ,palette_areas)
axs[0,0].get_legend().remove()
dv = 'Ripple modulation (50-120 ms)'
ylabel = 'Late ripple modulation (50-120 ms)'
point_plot_modulation_ripples(summary_units_df_sub, dv,parent_area, order, filter_spiking,axs[0,1], ylabel,ylim,  palette_areas)


parent_area='MB'
order= ['APN', 'MB']

dv = 'Ripple modulation (0-50 ms)'
ylabel = 'Early ripple modulation (0-50 ms)'

point_plot_modulation_ripples(summary_units_df_sub, dv, parent_area, order,filter_spiking,axs[1, 0], ylabel,ylim, palette_areas)
axs[1, 0].get_legend().remove()
dv = 'Ripple modulation (50-120 ms)'
ylabel = 'Late ripple modulation (50-120 ms)'
point_plot_modulation_ripples(summary_units_df_sub, dv,parent_area, order, filter_spiking,axs[1, 1], ylabel,ylim, palette_areas)
axs[1, 1].get_legend().remove()


parent_area='TH'
order= ['LP', 'LGd', 'TH', 'LGv', 'SGN', 'PO']

dv = 'Ripple modulation (0-50 ms)'


point_plot_modulation_ripples(summary_units_df_sub, dv, parent_area, order,filter_spiking,axs[2, 0], ylabel,ylim, palette_areas)
axs[2,0].get_legend().remove()
dv = 'Ripple modulation (50-120 ms)'
ylabel = 'Late ripple modulation (50-120 ms)'
point_plot_modulation_ripples(summary_units_df_sub, dv,parent_area, order, filter_spiking,axs[2, 1], ylabel,ylim, palette_areas)
axs[2,1].get_legend().remove()

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).axes[0].set_position([0.069767, 0.698102, 0.388947, 0.266644])
plt.figure(1).axes[1].set_position([0.536503, 0.698102, 0.388947, 0.266644])
plt.figure(1).axes[2].set_position([0.068114, 0.378129, 0.388947, 0.266644])
plt.figure(1).axes[3].set_position([0.534850, 0.378129, 0.388947, 0.266644])
plt.figure(1).axes[4].set_position([0.068114, 0.058156, 0.388947, 0.266644])
plt.figure(1).axes[5].set_position([0.534850, 0.058156, 0.388947, 0.266644])
plt.figure(1).text(0.03892215568862286, 0.9735792622133594, 'A', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.48952095808383245, 0.9735792622133592, 'B', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[1].new
#% end: automatic generated code from pylustrator
#plt.show()

plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_17", dpi=300)