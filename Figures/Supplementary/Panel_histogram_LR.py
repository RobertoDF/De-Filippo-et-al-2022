import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dill
import seaborn as sns
from Utils.Utils import clean_ripples_calculations
from Utils.Settings import output_folder_calculations, output_folder_figures_calculations, var_thr


with open(f"{output_folder_figures_calculations}/temp_for_supp_fig_ML_limits.pkl", "rb") as fp:  # We need this for a weird problem with pylustrator
    height1, height2, xs, ys, Q1, Q3 = dill.load(fp)

with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)


input_rip = []
for session_id in ripples_calcs.keys():
    ripples = ripples_calcs[session_id][3].copy()
    ripples = ripples[ripples["Area"] == "CA1"]
    ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)
    input_rip.append(ripples.groupby("Probe number-area").mean()["M-L (µm)"])

ml_space = pd.concat(input_rip)

medial_lim_ml = ml_space.quantile(.33333)
lateral_lim_ml = ml_space.quantile(.666666)

color_palette = sns.color_palette("flare", 255)

ax = sns.displot(data=ml_space, kde=True, bins=40, color="k", facecolor="w", edgecolor='black')

ax.axes[0, 0].fill_between(xs, 0, ys, where=(Q1 >= xs), interpolate=True, facecolor=color_palette[0], alpha=0.5, zorder=10)
ax.axes[0, 0].fill_between(xs, 0, ys, where=(Q3 <= xs), interpolate=True, facecolor=color_palette[254], alpha=0.5, zorder=10)
ax.axes[0, 0].fill_between(xs, 0, ys, where=((Q1 < xs) & (Q3 > xs)), interpolate=True,
                           facecolor=color_palette[int(254 / 2)], alpha=0.5, zorder=10)

ax.axes[0, 0].set_xlabel("M-L (µm)");

ax.axes[0, 0].text(.1, 0.9, 'Medial', transform=ax.axes[0, 0].transAxes, weight="bold", color=color_palette[0]);
ax.axes[0, 0].text(.7, 0.9, 'Lateral', transform=ax.axes[0, 0].transAxes, weight="bold", color=color_palette[254]);
ax.axes[0, 0].text(.4, 0.9, 'Central', transform=ax.axes[0, 0].transAxes, weight="bold",
                   color=color_palette[int(254 / 2)]);

plt.show()