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

fig, axs = plt.subplots(1, 2, figsize=(16, 8))

data = pd.melt(tot_summary_early[["Lateral seed", "Medial seed", "Session id"]],
               value_vars=["Lateral seed", "Medial seed"],
               id_vars=["Session id"], var_name='Post ripple phase', value_name='Spiking rate per 10 ms').groupby(
    ["Session id", "Post ripple phase"]).mean().reset_index()

sns.boxplot(data=data, x="Post ripple phase", y="Spiking rate per 10 ms", showfliers=False, ax=axs[0], palette=palette_ML,
            order=["Medial seed", "Lateral seed"])
ax = sns.stripplot(data=tot_summary_early.groupby("Session id")[["Lateral seed", "Medial seed"]].mean(), ax=axs[0],
                   dodge=True, size=2, color=".9", linewidth=0.6, jitter=0.1, order=["Medial seed", "Lateral seed"])
axs[0].set_title("Early phase (0-50 ms)")

print(pg.normality(data, dv="Spiking rate per 10 ms", group="Post ripple phase"))
print(pg.ttest(data[data["Post ripple phase"] == "Medial seed"]["Spiking rate per 10 ms"],
               data[data["Post ripple phase"] == "Lateral seed"]["Spiking rate per 10 ms"])["p-val"])
print(pg.mwu(data[data["Post ripple phase"] == "Medial seed"]["Spiking rate per 10 ms"],
             data[data["Post ripple phase"] == "Lateral seed"]["Spiking rate per 10 ms"])["p-val"])

ttest_early_spiking = pg.ttest(data[data["Post ripple phase"] == "Medial seed"]["Spiking rate per 10 ms"],
                  data[data["Post ripple phase"] == "Lateral seed"]["Spiking rate per 10 ms"], paired=True)

if ttest_early_spiking["p-val"].values[0] < 0.05:
    annot = Annotator(ax, data=data, pairs=[("Medial seed", "Lateral seed")],
                      y="Spiking rate per 10 ms", x="Post ripple phase", palette=palette_ML,
                      order=["Medial seed", "Lateral seed"])
    (annot
     .configure(test=None, test_short_name="custom test", text_format='star', loc='inside', verbose=0)
     .set_pvalues(pvalues=ttest_early_spiking["p-val"])
     .set_custom_annotations(["*"] * ttest_early_spiking["p-val"].shape[0])
     .annotate())
    axs[0].text(.4, .7, "Cohen's d: " + str(round(ttest_early_spiking["cohen-d"].values[0], 2)), transform=axs[0].transAxes,
                fontsize=6, ha='center', va='center');

data = pd.melt(tot_summary_late[["Lateral seed", "Medial seed", "Session id"]],
               value_vars=["Lateral seed", "Medial seed"], id_vars=["Session id"], var_name='Post ripple phase',
               value_name='Spiking rate per 10 ms').groupby(["Session id", "Post ripple phase"]).mean().reset_index()

sns.boxplot(data=data, x="Post ripple phase", y="Spiking rate per 10 ms", showfliers=False, ax=axs[1], palette=palette_ML,
            order=["Medial seed", "Lateral seed"])
ax = sns.stripplot(data=tot_summary_late.groupby("Session id")[["Lateral seed", "Medial seed"]].mean(), ax=axs[1],
                   dodge=True, size=2, color=".9", linewidth=0.6, jitter=0.1, order=["Medial seed", "Lateral seed"])

print(pg.normality(data, dv="Spiking rate per 10 ms", group="Post ripple phase"))
print(pg.ttest(data[data["Post ripple phase"] == "Medial seed"]["Spiking rate per 10 ms"],
               data[data["Post ripple phase"] == "Lateral seed"]["Spiking rate per 10 ms"])["p-val"])
print(pg.mwu(data[data["Post ripple phase"] == "Medial seed"]["Spiking rate per 10 ms"],
             data[data["Post ripple phase"] == "Lateral seed"]["Spiking rate per 10 ms"])["p-val"])

ttest_late_spiking = pg.ttest(data[data["Post ripple phase"] == "Medial seed"]["Spiking rate per 10 ms"],
                  data[data["Post ripple phase"] == "Lateral seed"]["Spiking rate per 10 ms"], paired=True)

if ttest_late_spiking["p-val"].values[0] < 0.05:
    annot = Annotator(ax, data=data, pairs=[("Medial seed", "Lateral seed")],
                      y="Spiking rate per 10 ms", x="Post ripple phase", palette=palette_ML,
                      order=["Medial seed", "Lateral seed"])
    (annot
     .configure(test=None, test_short_name="custom test", text_format='star', loc='inside', verbose=0)
     .set_pvalues(pvalues=ttest_late_spiking["p-val"])
     .set_custom_annotations(["*"] * ttest_late_spiking["p-val"].shape[0])
     .annotate())
    axs[1].text(.6, .7, "Cohen's d: " + str(round(ttest_late_spiking["cohen-d"].values[0], 2)), transform=axs[1].transAxes,
                fontsize=6, ha='center', va='center');

axs[1].set_title("Late phase (50-120 ms)")

plt.show()
