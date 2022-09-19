import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from Utils.Utils import color_to_labels, acronym_color_map
from Utils.Settings import output_folder_figures_calculations, output_folder_supplementary, Adapt_for_Nature_style
from Utils.Utils import Naturize

# pylustrator.start()

sns.set_theme(context='paper', style="ticks", rc={"ytick.major.pad": 1, "axes.labelpad":-1,  "xtick.major.pad": 3,'axes.spines.right': False, 'axes.spines.top': False, "lines.linewidth": 0.5, "xtick.labelsize": 5, "ytick.labelsize": 5
                                                  , "axes.labelsize": 6, "xtick.major.size": 1, "ytick.major.size": 1 , "axes.titlesize" : 6})

my_colors = [ "#301A4B", "#087E8B", "#F59A8C","#5FAD56","#AFA2FF","#FAC748"]
# Set your custom color palette
my_palette = sns.set_palette(sns.color_palette(my_colors))


with open(f"{output_folder_figures_calculations}/temp_data_supp_figure_1.pkl", "rb") as fp:  # Unpickling
    data_area, ripple_power, session_id, session_summary, summary_table, ripples = pickle.load(fp)


fig, axs = plt.subplots(2, 1, figsize=(16,8))
colors = [rgb2hex([x / 255 for x in acronym_color_map.get(area.split("-")[0])]) for area in ripple_power.columns.get_level_values(1)]
g = sns.violinplot(data=ripple_power, scale="count", palette=colors, ax=axs[0])
#g.set_xticklabels([str(n) +"-" + str(area) for n, area in zip(ripple_power.columns.get_level_values(0), ripple_power.columns.get_level_values(1))], size=8);
g.set_xticklabels("");
g.set_ylabel("∫Ripple (mV*s)",color="#FE4A49");

colors = [rgb2hex([x / 255 for x in acronym_color_map.get(area.split("-")[0])]) for area in data_area.columns.get_level_values(1)]
g = sns.violinplot(data=data_area, scale="count", palette=colors, ax=axs[1])
#g.set_xticklabels([str(n) +"-" + str(area) for n, area in zip(ripple_power.columns.get_level_values(0), ripple_power.columns.get_level_values(1))], size=8);
g.set_xticklabels([str(n) +"-"+ str(area) for n, area in zip(data_area.columns.get_level_values(0), data_area.columns.get_level_values(1))], rotation=75);
color_to_labels(g, "x", "major", 1)
g.set_ylabel("RIVD (mV*s)", color= "#86A2BA");
g.set_xlabel("Probe number - area");
axs[0].set_title(f"Session {session_id}")

plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_violinplot_session", dpi=300)