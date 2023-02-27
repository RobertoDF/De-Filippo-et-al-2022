import pickle
import matplotlib.pyplot as plt
import numpy as np
from Utils.Settings import output_folder_figures_calculations

with open(f"{output_folder_figures_calculations}/temp_data_figure_1.pkl", "rb") as fp:  # Unpickling
    sessions, high_distance, low_distance, ripples_lags, ripples_lags_inverted_reference,  ripples_calcs, summary_corrs, distance_tabs = pickle.load(fp)

df = distance_tabs.corr()**2
fig, ax = plt.subplots(1, 1)
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(df.keys())))
ax.table(cellText=df.values, colLabels=df.keys(), rowLabels=df.keys(), loc='center', colColours=colors, rowColours=colors)
plt.axis("off")
plt.show()
