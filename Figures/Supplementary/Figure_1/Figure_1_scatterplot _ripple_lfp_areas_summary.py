import Utils.Style
from Utils.Style import palette_figure_1
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import dill
from Utils.Settings import output_folder_figures_calculations

with open(f"{output_folder_figures_calculations}/temp_data_supp_figure_1.pkl", 'rb') as f:
    data_area, ripple_power, session_id, session_summary, summary_table, ripples = dill.load(f)

high_var_areas = summary_table[summary_table["Count"] > 15].groupby("Area").std()[["μ(Z-scored ∫Ripple)", "μ(Z-scored RIVD)"]].sum(axis=1).sort_values(ascending=False)

high_var_areas = high_var_areas[high_var_areas > high_var_areas.std()*2.5].index

fig, ax = plt.subplots(figsize=(6, 6))

sns.scatterplot(data=summary_table[~summary_table["Area"].isin(high_var_areas)], x="μ(∫Ripple)", y="μ(RIVD)", s=10,
                color=(0.6, 0.6, 0.6), alpha=0.1, ax=ax)
g = sns.scatterplot(data=summary_table[summary_table["Area"].isin(high_var_areas)], x="μ(∫Ripple)", y="μ(RIVD)", s=10,
                    hue="Area", alpha=0.6, ax=ax, hue_order=high_var_areas, palette=palette_figure_1)
g.legend(loc='lower right', ncol=3, frameon=False, handletextpad=0.01, columnspacing=-0.2)


for area in high_var_areas:
    subdata = summary_table[summary_table["Area"] == area]
    y = subdata["μ(RIVD)"]
    x = subdata["μ(∫Ripple)"]
    fit = np.polyfit(x, y, deg=1)
    predict = np.poly1d(fit)

    color = palette_figure_1.get(area)
    # color = [[x / 255 for x in acronym_color_map.get(area.split("-")[0])]] * subdata.shape[0]
    ax.plot(x, predict(x), color=color, alpha=0.9, linewidth=1)

left, bottom, width, height = [0.78, 0.45, 0.1, 0.44]
ax2 = fig.add_axes([left, bottom, width, height])

colors = [palette_figure_1.get(area) for area in (
    summary_table[summary_table["Count"] > 15].groupby("Area").mean()[["μ(Z-scored ∫Ripple)", "μ(Z-scored RIVD)"]].sum(
        axis=1).sort_values(ascending=False).index)]
colors = [(0.6, 0.6, 0.6, 1) if v is None else v for v in colors]

_ = summary_table[summary_table["Count"] > 15].groupby("Area").mean()[["μ(Z-scored ∫Ripple)", "μ(Z-scored RIVD)"]].sum(
    axis=1).sort_values(ascending=False)[:10]
ax2.barh(y=_.index, width=_, color=colors)

ax2.yaxis.labelpad = 0
ax2.set_xlabel("Σμ");

plt.show()
