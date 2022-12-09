import dill
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import Utils.Style
import pingouin as pg
from Utils.Style import palette_ML
from Utils.Settings import output_folder_figures_calculations, output_folder_supplementary, Adapt_for_Nature_style
from Utils.Utils import Naturize
from statannotations.Annotator import Annotator
import pylustrator

pylustrator.start()

with open(f"{output_folder_figures_calculations}/temp_data_figure_4.pkl", 'rb') as f:
    space_sub_spike_times, target_area, units, field_to_use_to_compare, \
    session_id_example, lfp, lfp_per_probe, \
    ripple_cluster_lateral_seed, ripple_cluster_medial_seed, source_area, ripples, \
    tot_summary_early, summary_fraction_active_clusters_per_ripples_early, \
    summary_fraction_active_clusters_per_ripples_early_by_neuron_type, \
    tot_summary_late, summary_fraction_active_clusters_per_ripples_late, \
    summary_fraction_active_clusters_per_ripples_late_by_neuron_type, \
    tot_summary, summary_fraction_active_clusters_per_ripples, \
    summary_fraction_active_clusters_per_ripples_by_neuron_type = dill.load(f)

fig, axs = plt.subplots(1,2,figsize=(16,8))
plt.figure(1).set_size_inches((8.900000/2.54)*2, 8.900000/2.54, forward=True)
data=summary_fraction_active_clusters_per_ripples.groupby(["Session id", "Location seed"]).mean().reset_index()
data["Fraction active neurons per ripple (%)"] = data["Fraction active neurons per ripple (%)"]  * 100
sns.boxplot(data=data, x="Location seed", y="Fraction active neurons per ripple (%)", ax=axs[0],showfliers=False,  palette=palette_ML, order=["Medial seed", "Lateral seed"])
ax = sns.stripplot(data=data, x="Location seed", y="Fraction active neurons per ripple (%)",  ax=axs[0],  dodge=True, size=2, color=".9", linewidth=0.6, jitter=0.1, order=["Medial seed", "Lateral seed"])
axs[0].set_title("Early phase (0-50 ms)")


print(pg.normality(data, dv="Fraction active neurons per ripple (%)", group="Location seed"))
print(pg.ttest(data[data["Location seed"]=="Medial seed"]["Fraction active neurons per ripple (%)"], data[data["Location seed"]=="Lateral seed"]["Fraction active neurons per ripple (%)"], paired=True)["p-val"])
print(pg.mwu(data[data["Location seed"]=="Medial seed"]["Fraction active neurons per ripple (%)"], data[data["Location seed"]=="Lateral seed"]["Fraction active neurons per ripple (%)"])["p-val"])

ttest_clus_per_ripple = pg.ttest(data[data["Location seed"]=="Medial seed"]["Fraction active neurons per ripple (%)"], data[data["Location seed"]=="Lateral seed"]["Fraction active neurons per ripple (%)"], paired=True)[["p-val", "cohen-d"]]


if ttest_clus_per_ripple["p-val"].values[0]<0.05:
    annot = Annotator(ax, data=data, pairs= [("Medial seed", "Lateral seed")],
                       y="Fraction active neurons per ripple (%)", x="Location seed", palette=palette_ML,  order=["Medial seed", "Lateral seed"])
    (annot
     .configure(test=None, test_short_name="custom test",  text_format='star', loc='inside', verbose=0)
     .set_pvalues(pvalues=ttest_clus_per_ripple["p-val"])
     .set_custom_annotations(["*"]*ttest_clus_per_ripple["p-val"].shape[0])
     .annotate())
    axs[0].text(.5, .7, "Cohen's d: " + str(round(ttest_clus_per_ripple["cohen-d"].values[0], 2)), transform=axs[0].transAxes,
                fontsize=6, ha='center', va='center');

axs[0].set_title("Fraction active clusters")

data = pd.melt(tot_summary[["Lateral seed", "Medial seed", "Session id"]],
               value_vars=["Lateral seed", "Medial seed"], id_vars=["Session id"], var_name='Location seed',
               value_name='Spiking rate per 10 ms').groupby(["Session id", "Location seed"]).mean().reset_index()

sns.boxplot(data=data, x="Location seed", y="Spiking rate per 10 ms", showfliers=False, ax=axs[1], palette=palette_ML,
            order=["Medial seed", "Lateral seed"])
ax = sns.stripplot(data=tot_summary.groupby("Session id")[["Lateral seed", "Medial seed"]].mean(), ax=axs[1],
                   dodge=True, size=2, color=".9", linewidth=0.6, jitter=0.1, order=["Medial seed", "Lateral seed"])

print(pg.normality(data, dv="Spiking rate per 10 ms", group="Location seed"))
print(pg.ttest(data[data["Location seed"] == "Medial seed"]["Spiking rate per 10 ms"],
               data[data["Location seed"] == "Lateral seed"]["Spiking rate per 10 ms"])["p-val"])
print(pg.mwu(data[data["Location seed"] == "Medial seed"]["Spiking rate per 10 ms"],
             data[data["Location seed"] == "Lateral seed"]["Spiking rate per 10 ms"])["p-val"])

ttest_late_spiking = pg.ttest(data[data["Location seed"] == "Medial seed"]["Spiking rate per 10 ms"],
                  data[data["Location seed"] == "Lateral seed"]["Spiking rate per 10 ms"], paired=True)

if ttest_late_spiking["p-val"].values[0] < 0.05:
    annot = Annotator(ax, data=data, pairs=[("Medial seed", "Lateral seed")],
                      y="Spiking rate per 10 ms", x="Location seed", palette=palette_ML,
                      order=["Medial seed", "Lateral seed"])
    (annot
     .configure(test=None, test_short_name="custom test", text_format='star', loc='inside', verbose=0)
     .set_pvalues(pvalues=ttest_late_spiking["p-val"])
     .set_custom_annotations(["*"] * ttest_late_spiking["p-val"].shape[0])
     .annotate())
    axs[1].text(.5, .7, "Cohen's d: " + str(round(ttest_late_spiking["cohen-d"].values[0], 2)), transform=axs[1].transAxes,
                fontsize=6, ha='center', va='center');

axs[1].set_title("Spiking rate")

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.072857, 0.931429])
plt.figure(1).texts[0].set_text("A")
plt.figure(1).texts[0].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.494286, 0.931429])
plt.figure(1).texts[1].set_text("B")
plt.figure(1).texts[1].set_weight("bold")

if Adapt_for_Nature_style is True:
    Naturize()

#% end: automatic generated code from pylustrator
#plt.show()
plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_11", dpi=300)
