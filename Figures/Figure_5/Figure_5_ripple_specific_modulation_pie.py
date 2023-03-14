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

_ = pd.DataFrame(summary_units_df_sub[(summary_units_df_sub['Firing rate (0-120 ms) medial']>0.025)&
                     (summary_units_df_sub['Parent brain region']=='HPF' )]['Ripple type engagement'].value_counts())


def my_autopct(pct):
    return ('%.2f' % pct + " %") if pct > 20 else ''

fig, ax = plt.subplots()
_, _, autopcts = ax.pie(_.values.squeeze(),colors=[palette_ML[key] for key in _.index], autopct=my_autopct)

plt.setp(autopcts, **{'color':'white', 'weight':'bold', 'fontsize':12.5})

ax.xaxis.set_label_position('top')
ax.set_title('Ripple type engagement')

plt.show()