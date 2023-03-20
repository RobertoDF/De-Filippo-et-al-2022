import dill
import seaborn as sns
import matplotlib.pyplot as plt
from Utils.Settings import output_folder_figures_calculations
from Utils.Utils import  acronym_color_map
import pandas as pd
import Utils.Style
import scipy as sp

with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)

palette_parent_areas = dict()
for area in summary_units_df_sub['Parent brain region'].unique():
    palette_parent_areas[area] = '#' + ''.join(f'{i:02X}' for i in acronym_color_map[area]);


g = sns.FacetGrid(summary_units_df_sub, col='Parent brain region', hue='Parent brain region',
                  palette=palette_parent_areas, col_order=['HPF', 'Isocortex', 'MB', 'TH'])
g.map_dataframe(sns.regplot, x='Firing rate (0-120 ms) medial', y='Firing rate (120-0 ms) medial', scatter_kws=dict(s=1, alpha=.3))
g.map_dataframe(lambda data, **kws: plt.axline((0, 0), slope=1, alpha=.5, linestyle='--', color='k'))


def annotate(data, **kws):
    g = data['Parent brain region'].unique()[0]

    # get the y-position from the dict
    y = .8

    r, p = sp.stats.pearsonr(data['Firing rate (0-120 ms) medial'], data['Firing rate (120-0 ms) medial'])
    ax = plt.gca()
    if p == 0:
        ax.text(.1, y, f'{g}: R²={r ** 2:.2f}, p<0.0005', transform=ax.transAxes, color=palette_parent_areas[g])
    else:
        ax.text(.1, y, f'{g}: R²={r ** 2:.2f}, p={2:.2e}', transform=ax.transAxes, color=palette_parent_areas[g])


g.map_dataframe(annotate)


def plot_diag(data, **kws):
    ax = plt.gca()
    ax.plot([0, ax.get_ylim()[1]], [0, ax.get_ylim()[1] / (5/3)], alpha=.5, linestyle='--', color='r')
    #ax.plot([0, ax.get_ylim()[1] / 2], [0, ax.get_ylim()[1]], alpha=.5, linestyle='--', color='r')


g.map_dataframe(plot_diag)
plt.show()