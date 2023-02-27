import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import Utils.Style
from Utils.Settings import output_folder_figures_calculations

with open(f"{output_folder_figures_calculations}/temp_data_figure_1.pkl", "rb") as fp:  # Unpickling
    sessions, high_distance, low_distance, ripples_lags, ripples_lags_inverted_reference, ripples_calcs, summary_corrs, distance_tabs = pickle.load(fp)

out = []

for session_id, sel in ripples_calcs.items():

    ripple_power = sel[0][0].copy()

    ripple_power = ripple_power.loc[:, ripple_power.var() > 5]

    ripple_power = ripple_power[ripple_power.columns[ripple_power.columns.get_level_values(1) == "CA1"]]
    ripple_power.columns = ["0-CA1", "1-CA1", "2-CA1", "3-CA1", "4-CA1", "5-CA1"][:ripple_power.columns.shape[0]]
    if ripple_power.shape[1] > 0:

        aa = ripple_power.corr().stack().reset_index(name="Correlation")
        aa.columns = ["Primary area", "Secondary area", "Correlation"]
        aa["Session"] = session_id

        out.append(aa)

out2 = [i for i in out if i.shape[0] >= 25]  # I want 5 ca1 regions for nice looking matrix

out_corrs = pd.concat(list( out2[i] for i in [1, 3, 6, 9]))  # select examples

corr_mat = out_corrs
with sns.axes_style("whitegrid"):
    g = sns.relplot(data=corr_mat,
                    x="Primary area", y="Secondary area", hue="Correlation", size="Correlation",
                    palette="mako", hue_norm=(0.5, 1), edgecolor="0",
                    sizes=(5, 75), size_norm=(0.5, 1), col="Session", height=1.5, aspect=8/8, col_wrap=2)

    g.set(xlabel="", ylabel="")
    g.despine(left=True, bottom=True)
    for q in range(out_corrs.shape[1]):
        g.axes[q].margins(0.2)

    for artist in g.legend.legendHandles:
        artist.set_edgecolor("0")
    g.legend.columnspacing=8
    leg = g._legend
    #leg._loc = 2
    leg.set_bbox_to_anchor([0.1, 0.85])
    g.set_xticklabels(rotation=75)

plt.show()
