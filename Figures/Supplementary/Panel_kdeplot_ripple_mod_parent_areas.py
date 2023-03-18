import dill
import seaborn as sns
import matplotlib.pyplot as plt
from Utils.Settings import output_folder_figures_calculations, minimum_firing_rate_hz
from Utils.Utils import  palette_ML, plot_dist_ripple_mod
import pandas as pd
import Utils.Style
import pingouin as pg

with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)


def func_annotate(data, **kws):
    ax0 = plt.gca()
    # print(data["Parent brain region"].unique())
    norm_test = pg.normality(data=data, dv=param, group="Ripple seed")

    if norm_test["normal"].all():
        p_val = pg.ttest(data[data["Ripple seed"] == "Medial"][param],
                         data[data["Ripple seed"] == "Lateral"][param], paired=True)["p-val"][0]
        print("ttest: ", p_val)

    else:
        p_val = \
        pg.wilcoxon(data[data["Ripple seed"] == "Medial"][param], data[data["Ripple seed"] == "Lateral"][param])["p-val"][0]
        cles = \
        pg.wilcoxon(data[data["Ripple seed"] == "Medial"][param], data[data["Ripple seed"] == "Lateral"][param])["CLES"][0]
        print("mwu p-val and CLES: ", p_val, cles)

    if p_val < .05:
        ax0.text(.6, .8, "*",
                 transform=ax0.transAxes,
                 fontsize=15, ha='center', va='center');
        ax0.text(.6, .7, f"p-value = {'{:.2e}'.format(p_val)}",
                 transform=ax0.transAxes,
                 fontsize=10, ha='center', va='center');
        ax0.text(.6, .6, f"CLES = {round(cles, 3)}",
                 transform=ax0.transAxes,
                 fontsize=10, ha='center', va='center');


_ = summary_units_df_sub[(summary_units_df_sub['Firing rate (0-50 ms) medial']>minimum_firing_rate_hz) |
                                                        (summary_units_df_sub['Firing rate (0-50 ms) lateral']>minimum_firing_rate_hz)][['Ripple modulation (0-120 ms) medial', 'Ripple modulation (0-120 ms) lateral', 'Parent brain region']]

_ = pd.wide_to_long(_.reset_index(), stubnames='Ripple modulation (0-120 ms)', i=['Parent brain region','unit_id'], j="Ripple seed", sep=' ', suffix=r'\w+').reset_index()
_['Ripple seed'] = _["Ripple seed"].str.capitalize()
param = 'Ripple modulation (0-120 ms)'
g = sns.FacetGrid(_[_['Parent brain region']!='HPF'], col='Parent brain region', sharex=False, sharey=False, col_order=['Isocortex', 'MB', 'TH'])
g.map_dataframe(sns.kdeplot, x='Ripple modulation (0-120 ms)', hue='Ripple seed', palette=palette_ML, fill=True, gridsize=500)
g.refline(x=0)
g.refline(x=1, color='r')
g.set(xlim=(-1, 3))
g.set(yticks=[])
g.map_dataframe(func_annotate)