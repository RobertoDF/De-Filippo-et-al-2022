import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import Utils.Style
from Utils.Style import palette_ML
from Utils.Settings import output_folder_figures_calculations
from statannotations.Annotator import Annotator

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
data = summary_fraction_active_clusters_per_ripples_early.groupby(["Session id", "Location seed"]).mean().reset_index()
data["Fraction active neurons per ripple (%)"] = data["Fraction active neurons per ripple (%)"]  * 100
sns.boxplot(data=data, x="Location seed", y="Fraction active neurons per ripple (%)", ax=axs[0],showfliers=False,  palette=palette_ML, order=["Medial seed", "Lateral seed"])
ax = sns.stripplot(data=data, x="Location seed", y="Fraction active neurons per ripple (%)",  ax=axs[0],  dodge=True, size=2, color=".9", linewidth=0.6, jitter=0.1, order=["Medial seed", "Lateral seed"])
axs[0].set_title("Early phase (0-50 ms)")


print(pg.normality(data, dv="Fraction active neurons per ripple (%)", group="Location seed"))
print(pg.ttest(data[data["Location seed"]=="Medial seed"]["Fraction active neurons per ripple (%)"], data[data["Location seed"]=="Lateral seed"]["Fraction active neurons per ripple (%)"], paired=True)["p-val"])
print(pg.mwu(data[data["Location seed"]=="Medial seed"]["Fraction active neurons per ripple (%)"], data[data["Location seed"]=="Lateral seed"]["Fraction active neurons per ripple (%)"])["p-val"])

ttest_early_clus_per_ripple = pg.ttest(data[data["Location seed"]=="Medial seed"]["Fraction active neurons per ripple (%)"], data[data["Location seed"]=="Lateral seed"]["Fraction active neurons per ripple (%)"], paired=True)[["p-val", "cohen-d"]]


if ttest_early_clus_per_ripple["p-val"].values[0]<0.05:
    annot = Annotator(ax, data=data, pairs= [("Medial seed", "Lateral seed")],
                       y="Fraction active neurons per ripple (%)", x="Location seed", palette=palette_ML,  order=["Medial seed", "Lateral seed"])
    (annot
     .configure(test=None, test_short_name="custom test",  text_format='star', loc='inside', verbose=0)
     .set_pvalues(pvalues=ttest_early_clus_per_ripple["p-val"])
     .set_custom_annotations(["*"]*ttest_early_clus_per_ripple["p-val"].shape[0])
     .annotate())
    axs[0].text(.4, .7, "Cohen's d: " + str(round(ttest_early_clus_per_ripple["cohen-d"].values[0], 2)), transform=axs[0].transAxes,
                fontsize=6, ha='center', va='center');



data = summary_fraction_active_clusters_per_ripples_late.groupby(["Session id", "Location seed"]).mean().reset_index()
data["Fraction active neurons per ripple (%)"] = data["Fraction active neurons per ripple (%)"]  * 100
sns.boxplot(data=data, x="Location seed", y="Fraction active neurons per ripple (%)", ax=axs[1], showfliers=False, palette=palette_ML, order=["Medial seed", "Lateral seed"])
ax = sns.stripplot(data=data, x="Location seed", y="Fraction active neurons per ripple (%)",  ax=axs[1],  dodge=True, size=2, color=".9", linewidth=0.6, jitter=0.1, order=["Medial seed", "Lateral seed"])
axs[1].set_title("Late phase (50-120 ms)")

print(pg.normality(data, dv="Fraction active neurons per ripple (%)", group="Location seed"))
print(pg.ttest(data[data["Location seed"]=="Medial seed"]["Fraction active neurons per ripple (%)"], data[data["Location seed"]=="Lateral seed"]["Fraction active neurons per ripple (%)"], paired=True)["p-val"])
print(pg.mwu(data[data["Location seed"]=="Medial seed"]["Fraction active neurons per ripple (%)"], data[data["Location seed"]=="Lateral seed"]["Fraction active neurons per ripple (%)"])["p-val"])

ttest_late_clus_per_ripple = pg.ttest(data[data["Location seed"]=="Medial seed"]["Fraction active neurons per ripple (%)"], data[data["Location seed"]=="Lateral seed"]["Fraction active neurons per ripple (%)"])


if ttest_late_clus_per_ripple["p-val"].values[0]<0.05:
    annot = Annotator(ax, data=data, pairs= [("Medial seed", "Lateral seed")],
                       y="Fraction active neurons per ripple (%)", x="Location seed", palette=palette_ML,  order=["Medial seed", "Lateral seed"])
    (annot
     .configure(test=None, test_short_name="custom test",  text_format='star', loc='inside', verbose=0)
     .set_pvalues(pvalues=ttest_late_clus_per_ripple["p-val"])
     .set_custom_annotations(["*"]*ttest_late_clus_per_ripple["p-val"].shape[0])
     .annotate())
    axs[1].text(.6, .7, "Cohen's d: " + str(round(ttest_late_clus_per_ripple["cohen-d"].values[0], 2)), transform=axs[1].transAxes,
                   fontsize=6, ha='center', va='center');

plt.show()
