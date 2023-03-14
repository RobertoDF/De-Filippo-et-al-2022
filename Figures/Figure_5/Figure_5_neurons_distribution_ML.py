import dill
import seaborn as sns
import matplotlib.pyplot as plt
from Utils.Settings import output_folder_figures_calculations, output_folder_calculations,  var_thr
from Utils.Style import palette_ML, palette_HPF
import pandas as pd
import Utils.Style

with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)

with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)


spike_hists = {}

input_rip = []
for session_id in ripples_calcs.keys():
    ripples = ripples_calcs[session_id][3].copy()
    ripples = ripples.groupby('Probe number-area').filter(lambda group: group['∫Ripple'].var() > var_thr)
    input_rip.append(ripples.groupby('Probe number-area').mean()['L-R (µm)'])

lr_space = pd.concat(input_rip)

medial_lim = lr_space.quantile(.33333)
lateral_lim = lr_space.quantile(.666666)
center = lr_space.median()
medial_lim_lm = medial_lim - 5691.510009765625
lateral_lim_lm = lateral_lim - 5691.510009765625

def l_m_classifier(row):
    if row['Source M-L (µm)'] < medial_lim_lm:
        v = 'Medial'
    elif row['Source M-L (µm)'] > lateral_lim_lm:
        v = 'Lateral'
    else:
        v = 'Central'
    return v

fig, axs = plt.subplots(2, figsize=(10,5),  sharex=True)
sns.kdeplot(ax=axs[0], data=summary_units_df_sub[ summary_units_df_sub['Parent brain region']=='HPF'],
            hue='Brain region', x='M-L',  palette=palette_HPF, fill=True,  alpha=.15, hue_order=['CA1',  'CA3', 'DG', 'ProS', 'SUB'], legend=False)
sns.kdeplot(ax=axs[1], data=summary_units_df_sub[ summary_units_df_sub['Parent brain region']=='HPF'],
            hue='Brain region', x='M-L',  palette=palette_HPF, multiple='stack', hue_order=['CA1',  'CA3', 'DG', 'ProS', 'SUB'])
axs[0].axvline(medial_lim - 5691.510009765625,  color= palette_ML["Medial"], linestyle='--')
axs[0].axvline(lateral_lim - 5691.510009765625,  color= palette_ML["Lateral"], linestyle='--')
axs[0].get_yaxis().set_visible(False)
axs[1].get_yaxis().set_visible(False)
axs[0].spines[['left']].set_visible(False)
axs[1].spines[['left']].set_visible(False)
plt.show()