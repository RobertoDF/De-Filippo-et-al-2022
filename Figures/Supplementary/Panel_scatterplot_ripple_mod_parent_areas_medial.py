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
def plot_func(*args, **kwargs):
    data = kwargs.pop('data')
    x = kwargs.pop('x')
    y = kwargs.pop('y')
    scatter_kws = kwargs.pop('scatter_kws')
    color = kwargs.pop('color')
    label = kwargs.pop('label')

    if (label=="Isocortex") | (label=="MB"):
        sns.regplot(data=data[data["Ripple engagement"]!='Ripple engagement'], x=x, y=y, scatter_kws=scatter_kws, color=color, label=label)
        sns.scatterplot(data=data[data["Ripple engagement"]=='Ripple engagement'], x=x, y=y, s=5,alpha=.3, color=".15")
    else:

        sns.regplot(data=data, x=x, y=y, scatter_kws=scatter_kws, color=color, label=label )


def annotate(data, **kws):
    g = data['Parent brain region'].unique()[0]

    if ((data['Parent brain region'] == "Isocortex").all()) | ((data['Parent brain region'] == "MB").all()):
        data = data[data["Ripple engagement"]!='Ripple engagement']

    # get the y-position from the dict
    y = .8

    r, p = sp.stats.pearsonr(data['Firing rate (0-50 ms) medial'], data['Firing rate (120-0 ms) medial'])
    ax = plt.gca()
    if p <0.001:
        ax.text(.1, y, f'{g}: R²={r ** 2:.2f}, p-value<0.001', transform=ax.transAxes, color=palette_parent_areas[g])
    else:
        ax.text(.1, y, f'{g}: R²={r ** 2:.2f}, p-value={"{:.2e}".format(p)}', transform=ax.transAxes, color=palette_parent_areas[g])


g.map_dataframe(plot_func, x='Firing rate (0-50 ms) lateral', y='Firing rate (120-0 ms) lateral', scatter_kws=dict(s=1, alpha=.3))
g.map_dataframe(lambda data, **kws: plt.axline((0, 0), slope=1, alpha=.5, linestyle='--', color='k'))

g.map_dataframe(annotate)


def plot_diag(data, **kws):
    ax = plt.gca()
    ax.plot([0, ax.get_ylim()[1]], [0, ax.get_ylim()[1] / (5/3)], alpha=.5, linestyle='--', color='r')
    #ax.plot([0, ax.get_ylim()[1] / 2], [0, ax.get_ylim()[1]], alpha=.5, linestyle='--', color='r')


g.map_dataframe(plot_diag)
plt.show()