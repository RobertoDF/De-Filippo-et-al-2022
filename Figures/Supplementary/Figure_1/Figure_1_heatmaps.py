import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from Utils.Utils import color_to_labels
from Utils.Settings import output_folder_figures_calculations
import Utils.Style


with open(f"{output_folder_figures_calculations}/temp_data_supp_figure_1.pkl", "rb") as fp:  # Unpickling
    data_area, ripple_power, session_id, session_summary, summary_table, ripples = pickle.load(fp)

fig, axs = plt.subplots(1, 2, figsize=(3, 4.8))
fig.suptitle(f'Session = {session_id}', fontsize=6)
cbar_ax = fig.add_axes([.91, .3, .03, .4])
g = sns.heatmap(data=ripple_power.T, cmap="rocket", vmin=0, vmax=20, yticklabels=1, xticklabels=1000,
           cbar_ax=cbar_ax, ax=axs[0])
g.yaxis.labelpad = -10
cbar_ax.yaxis.set_tick_params(pad=1)

color_to_labels(axs[0], "y", "major", 1)

color = []
out = []
colors_for_plot = [axs[0].get_yticklabels()[0].get_color()]
for n, ytick in enumerate(axs[0].get_yticklabels()):
    color_now = ytick.get_color()

    out.append(n)
    colors_for_plot.append(color_now)

out[0] = 0.1
for height, color in zip(out, colors_for_plot):
    axs[0].hlines([height], *axs[0].get_xlim(), color=color, linewidth=0.4);

cbar_ax2 = fig.add_axes([.41, .3, .03, .4])
sns.heatmap(data=data_area.T, cmap="rocket", yticklabels=1, xticklabels=1000,
            cbar_ax=cbar_ax2, ax=axs[1])
cbar_ax2.yaxis.set_tick_params(pad=1)
color_to_labels(axs[1], "y", "major", 1)

color = []
out = []
colors_for_plot = [axs[1].get_yticklabels()[0].get_color()]
for n, ytick in enumerate(axs[1].get_yticklabels()):
    color_now = ytick.get_color()

    out.append(n)
    colors_for_plot.append(color_now)

out[0] = 0.1
for height, color in zip(out, colors_for_plot):
    axs[1].hlines([height], *axs[1].get_xlim(), color=color, linewidth=0.4);

axs[1].get_yaxis().set_visible(False)

axs[0].set_title("∫Ripple", color="#FE4A49")
axs[1].set_title("RIVD", color="#86A2BA")

plt.show()