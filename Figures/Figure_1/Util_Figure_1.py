import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Utils.Settings import output_folder_figures_calculations

print("running utils figure 1, necessary for conflict with pylustrator")

with open(f"{output_folder_figures_calculations}/temp_data_figure_1.pkl", "rb") as fp:  # Unpickling
    sessions, high_distance, low_distance, ripples_lags, ripples_lags_inverted_reference, ripples_calcs, summary_corrs, distance_tabs = pickle.load(fp)


quartiles = summary_corrs[summary_corrs["Comparison"] == "CA1-CA1"]["Correlation"].quantile([0.25, 0.5, 0.75])
ax = sns.displot(data=summary_corrs[summary_corrs["Comparison"] == "CA1-CA1"], x="Correlation", kde=True, rug=True,
                 color="k", facecolor="w", edgecolor='black')


kdeline = ax.axes[0, 0].lines[0]
Q1 = quartiles[0.25]
xs = kdeline.get_xdata()
ys = kdeline.get_ydata()
height1 = np.interp(Q1, xs, ys)

kdeline = ax.axes[0, 0].lines[0]
Q3 = quartiles[0.75]
height2 = np.interp(Q3, xs, ys)

with open("/alzheimer/Roberto/Allen_Institute/temp/temp_for_fig_1.pkl", "wb") as fp:
    pickle.dump([height1, height2, xs, ys, Q1, Q3], fp)

plt.close()

