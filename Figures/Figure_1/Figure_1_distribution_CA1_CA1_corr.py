import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt
import Utils.Style
from Utils.Settings import output_folder_figures_calculations

with open(f"{output_folder_figures_calculations}/temp_data_figure_1.pkl", "rb") as fp:  # Unpickling
    sessions, high_distance, low_distance, ripples_lags, ripples_lags_inverted_reference, ripples_calcs, summary_corrs = pickle.load(fp)

with open(f"{output_folder_figures_calculations}/temp_for_fig_1.pkl", "rb") as fp:  # We need this for a weird problem with pylustrator
    height1, height2, xs, ys, Q1, Q3 = pickle.load(fp)

quartiles = summary_corrs[summary_corrs["Comparison"] == "CA1-CA1"]["Correlation"].quantile([0.25, 0.5, 0.75])
ax = sns.displot(data=summary_corrs[summary_corrs["Comparison"] == "CA1-CA1"], x="Correlation", kde=True,
                 color="k", facecolor="w", edgecolor='black')

ax.axes[0, 0].vlines(Q1, 0, height1, color=sns.husl_palette(2)[0])
ax.axes[0, 0].vlines(Q3, 0, height2, color=sns.husl_palette(2)[1])

ax.axes[0, 0].fill_between(xs, 0, ys, where=(Q1 >= xs), interpolate=True, facecolor=sns.husl_palette(2)[0], alpha=0.6, zorder=10)
ax.axes[0, 0].fill_between(xs, 0, ys, where=(Q3 <= xs), interpolate=True, facecolor=sns.husl_palette(2)[1], alpha=0.6, zorder=10)

ax.axes[0, 0].set_xlabel("Correlation ∫Ripple CA1-CA1");

ax.axes[0, 0].text(.3, 0.9, '$\it{Q\u2081}$', transform=ax.axes[0, 0].transAxes, weight="bold", fontsize=6, color=sns.husl_palette(2)[0]);
ax.axes[0, 0].text(.93, 0.9, '$\it{Q\u2084}$', transform=ax.axes[0, 0].transAxes, weight="bold", fontsize=6,color=sns.husl_palette(2)[1]);
plt.show()