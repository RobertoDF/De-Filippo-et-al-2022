from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from Utils.Settings import output_folder_figures_calculations, output_folder_supplementary
from Utils.Utils import acronym_color_map
import seaborn as sns
import numpy as np
import pickle

with open(f"{output_folder_figures_calculations}/temp_data_supp_figure_1.pkl", "rb") as fp:  # Unpickling
    data_area, ripple_power, session_id, session_summary, summary_table, ripples = pickle.load(fp)

fig, axs = plt.subplots(1,2, figsize=(10,4))
colors = []
for area in summary_table["Area"]:
    color = [x / 255 for x in acronym_color_map.get(area.split("-")[0])]
    colors.append(color)

ax = sns.scatterplot(data=summary_table, x="μ(∫Ripple)", y="$σ^2$(∫Ripple)", c=colors, alpha=0.4, ax=axs[0])
ax.xaxis.label.set_color("#FE4A49")
ax.yaxis.label.set_color("#FE4A49")
summary_table.sort_values("μ(∫Ripple)", inplace=True)
x = summary_table["μ(∫Ripple)"]
y = summary_table["$σ^2$(∫Ripple)"]
fit = np.polyfit(x, y, deg=2)
predict = np.poly1d(fit)

axs[0].plot(x, predict(x), color="k", alpha=0.3)

# axs[0].set_xlim([0, 10])
# axs[0].set_ylim([0, 10])

ax.text(.1,.8,f"R\u00b2={round(r2_score(y, predict(x)),4)}", transform=axs[0].transAxes, fontsize=14, weight="bold");

ax = sns.scatterplot(data=summary_table, x="μ(RIVD)", y="$σ^2$(RIVD)", c=colors, alpha=0.4, ax=axs[1])
ax.xaxis.label.set_color("#86A2BA")
ax.yaxis.label.set_color("#86A2BA")

summary_table.sort_values("μ(RIVD)", inplace=True)
x = summary_table["μ(RIVD)"]
y = summary_table["$σ^2$(RIVD)"]
fit = np.polyfit(x, y, deg=2)
predict = np.poly1d(fit)

axs[1].plot( x, predict(x),color="k", alpha=0.3)

ax.text(.1, .8,f"R\u00b2={round(r2_score(y, predict(x)),4)}",transform=axs[1].transAxes, fontsize=14, weight="bold");

plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_variance_mean_ripple_strength", dpi=300)
