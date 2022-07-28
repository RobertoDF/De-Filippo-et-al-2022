import dill
import matplotlib.pyplot as plt
import Utils.Style
from Utils.Style import palette_figure_1
from Utils.Settings import output_folder_figures_calculations
import seaborn as sns


with open(f"{output_folder_figures_calculations}/temp_data_supp_figure_1.pkl", 'rb') as f:
    data_area, ripple_power, session_id, session_summary, summary_table, ripples = dill.load(f)

high_var_areas_sess = session_summary.groupby("Area").std()[["Z-scored ∫Ripple", "Z-scored RIVD"]].sum(axis=1).sort_values(ascending=False)
high_var_areas_sess = high_var_areas_sess[high_var_areas_sess > high_var_areas_sess.std()*2.5].index

fig, ax = plt.subplots(figsize=(5, 5))
g = sns.kdeplot(data=session_summary[session_summary["Area"].isin(high_var_areas_sess)], x="∫Ripple", y="RIVD",
              hue="Area",  hue_order=high_var_areas_sess, palette=palette_figure_1 , kde_kws={'clip': (0.0, 20.0)}, ax=ax)
sns.move_legend(ax, "lower right", ncol=3, frameon=False, handletextpad=0.01, handlelength=1,  columnspacing=0.1, title="")
g.set(ylim=(None, 150), xlim=(None, 20))

colors = [palette_figure_1.get(area) for area in session_summary.groupby("Area").mean()[["Z-scored ∫Ripple", "Z-scored RIVD"]].sum(axis=1).sort_values(ascending=False).index]
colors = [(0.55,0.6,0.65,1) if v is None else v for v in colors]

left, bottom, width, height = [0.78, 0.45, 0.1, 0.44]
ax2 = fig.add_axes([left, bottom, width, height])

_ = session_summary.groupby("Area").mean()[["Z-scored ∫Ripple",  "Z-scored RIVD"]].sum(axis=1).sort_values(ascending=False)[:10]
ax2.barh(y=_.index, width=_, color=colors)
#ax2.set_yticklabels(_.index, rotation=75);
ax2.yaxis.labelpad = 0
ax2.set_xlabel("Σµ");

plt.show()



