import dill
import matplotlib.pyplot as plt
from Utils.Settings import output_folder_figures_calculations, output_folder_supplementary
from Utils.Utils import acronym_color_map, point_plot_modulation_ripples
import Utils.Style
import pylustrator

pylustrator.start()

with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)

palette_areas = dict()
for area in summary_units_df_sub['Brain region'].unique():
    palette_areas[area] = '#' + ''.join(f'{i:02X}' for i in acronym_color_map[area]);
fig, axs = plt.subplots(1,3,  figsize=(15,5))

ylim = [-.25, .5]
parent_area='Isocortex'
order= ['VIS', 'VISam', 'VISpm', 'VISp', 'VISl', 'VISrl', 'VISal']

dv = 'Pre-ripple modulation'
ylabel = 'Pre-ripple modulation (20-0ms)'
filter_spiking = summary_units_df_sub['Firing rate (0-120 ms) medial']>0.025
point_plot_modulation_ripples(summary_units_df_sub, dv, parent_area, order,filter_spiking,axs[0], ylabel,ylim, palette_areas)

parent_area='MB'
order= ['APN', 'MB']

dv = 'Pre-ripple modulation'
ylabel = 'Pre-ripple modulation (20-0ms)'
filter_spiking = summary_units_df_sub['Firing rate (0-120 ms) medial']>0.025
point_plot_modulation_ripples(summary_units_df_sub, dv, parent_area, order,filter_spiking,axs[1], ylabel, ylim,palette_areas)

parent_area='TH'
order= ['LP', 'LGd', 'TH', 'LGv', 'SGN', 'Eth', 'PO']

dv = 'Pre-ripple modulation'
ylabel='Pre-ripple modulation (20-0ms)'
filter_spiking = summary_units_df_sub['Firing rate (0-120 ms) medial']>0.025
point_plot_modulation_ripples(summary_units_df_sub, dv, parent_area, order,filter_spiking,axs[2], ylabel, ylim,palette_areas)


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).axes[0].set_position([0.053459, 0.110000, 0.263781, 0.770000])
plt.figure(1).axes[1].set_position([0.369996, 0.110000, 0.263781, 0.770000])
plt.figure(1).axes[2].set_position([0.686533, 0.110000, 0.263781, 0.770000])
plt.figure(1).text(0.013364779874213846, 0.9339622641509431, 'A', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.34119496855345977, 0.9339622641509431, 'B', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[1].new
plt.figure(1).text(0.661949685534591, 0.9339622641509431, 'C', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[2].new
#% end: automatic generated code from pylustrator
#plt.show()

plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_18", dpi=300)