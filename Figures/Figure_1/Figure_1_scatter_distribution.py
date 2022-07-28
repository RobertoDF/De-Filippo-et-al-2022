import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import Utils.Style
from Utils.Settings import output_folder_figures_calculations

with open(f"{output_folder_figures_calculations}/temp_data_figure_1.pkl", "rb") as fp:  # Unpickling
    sessions, high_distance, low_distance, ripples_lags, ripples_lags_inverted_reference, ripples_calcs, summary_corrs = pickle.load(fp)

clip = (-100, 100)
ripples_lags = ripples_lags[ripples_lags["Lag (ms)"].between(clip[0], clip[1])]

fig, axs = plt.subplots(3, figsize=(8, 6))
sns.scatterplot(data=ripples_lags[(ripples_lags["Type"] == "High distance (µm)")], x="Lag (ms)", y="∫Ripple", color=sns.husl_palette(2)[0], s=0.1, alpha=0.5, ax=axs[0])#
axs[0].tick_params(labelbottom=False)
axs[0].xaxis.label.set_visible(False)
axs[1].xaxis.label.set_visible(False)
axs[1].tick_params(labelbottom=False)
sns.scatterplot(data=ripples_lags[(ripples_lags["Type"] == "Low distance (µm)")], x="Lag (ms)", y="∫Ripple", color=sns.husl_palette(2)[1], s=0.1, alpha=0.5, ax=axs[1])#

sns.kdeplot(data=ripples_lags, x="Lag (ms)", hue="Type", fill=True, common_norm=True, clip=(clip[0], clip[1]),
   alpha=.5, linewidth=0, multiple="layer", palette=[sns.husl_palette(2)[0], sns.husl_palette(2)[1]], ax=axs[2])#
plt.legend([], [], frameon=False)

plt.show()
