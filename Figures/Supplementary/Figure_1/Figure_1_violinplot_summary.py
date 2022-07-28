import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Utils.Utils import color_to_labels
from Utils.Settings import output_folder_figures_calculations
import Utils.Style

fig, axs = plt.subplots(figsize=(8, 4))

with open(f"{output_folder_figures_calculations}/temp_data_supp_figure_1.pkl", "rb") as fp:  # Unpickling
    data_area, ripple_power, session_id, session_summary, summary_table, ripples = pickle.load(fp)

min_count = 6
long_summary = pd.melt(summary_table, id_vars=["Area", "Count", "Graph order"], value_vars=["μ(Z-scored ∫Ripple)", "μ(Z-scored RIVD)"], var_name="Measure type", value_name="Σμ")
graph_order = long_summary[long_summary["Count"] > min_count].groupby("Area").mean().sort_values(by="Graph order").index

ax = sns.violinplot(data=long_summary[long_summary["Count"] > min_count], x="Area", y="Σμ", hue="Measure type", scale="width", fliersize=0.5, split=True, palette=["#FE4A49", "#86A2BA"], order=graph_order )
color_to_labels(ax, "x", "major", pos=0)
ax.legend(loc='upper right', frameon=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=75);

plt.show()
