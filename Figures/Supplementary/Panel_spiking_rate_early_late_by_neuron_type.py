import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import Utils.Style
from Utils.Utils import format_for_annotator
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


def neuron_classifier(row):
    if row["waveform_duration"] < .4:
        v = "Putative inh"
    elif row["waveform_duration"] >= .4:
        v = "Putative exc"
    return v

fig, axs = plt.subplots(1, 2, figsize=(16, 8))

PROPS = {
    'medianprops':{'color':'white'},
}
tot_summary_early["Neuron type"] = tot_summary_early.apply(neuron_classifier, axis=1)
tot_summary_late["Neuron type"] = tot_summary_late.apply(neuron_classifier, axis=1)


data = pd.melt(tot_summary_early[["Lateral seed", "Medial seed", "Session id", "Neuron type"]],
               value_vars=["Lateral seed", "Medial seed"],
               id_vars=["Session id", "Neuron type"], var_name='Location seed', value_name='Spiking rate per 10 ms').groupby(
    ["Neuron type","Session id", "Location seed"]).mean().reset_index()


hue = "Location seed"
y="Spiking rate per 10 ms"
x="Neuron type"
hue_order=["Medial seed", "Lateral seed"]
order=["Putative exc", "Putative inh"]
ax = sns.boxplot(data=data, x=x, y=y, hue=hue, showfliers=False, ax=axs[0], palette=palette_ML,
            order=order, hue_order=hue_order, **PROPS)

ax.get_legend().remove()

ax = sns.stripplot(data=data,  x=x, y=y, ax=axs[0],
                   hue=hue,  dodge=True, size=2, color=".9", linewidth=0.1, jitter=0.1,
                   order=order, hue_order=hue_order)

ax.get_legend().remove()



# When creating the legend, only use the first two elements
# to effectively remove the last two.


print(pg.normality(data, dv="Spiking rate per 10 ms", group="Location seed"))
print(pg.ttest(data[data["Location seed"] == "Medial seed"]["Spiking rate per 10 ms"],
               data[data["Location seed"] == "Lateral seed"]["Spiking rate per 10 ms"])["p-val"])
print(pg.mwu(data[data["Location seed"] == "Medial seed"]["Spiking rate per 10 ms"],
             data[data["Location seed"] == "Lateral seed"]["Spiking rate per 10 ms"])["p-val"])

out_test = pd.DataFrame(data[data["Location seed"]=="Medial seed"]\
            .pairwise_tukey(effsize="cohen", dv="Spiking rate per 10 ms", between="Neuron type")[["A","B", "p-tukey"]])

pairs, pvalues = format_for_annotator(out_test, "Location seed", "Medial seed")
if len(pairs)>0:
    pairs = [[(sub[1], sub[0]) for sub in pairs[0]]]

out_test = pd.DataFrame(data[data["Location seed"]=="Lateral seed"]\
            .pairwise_tukey(effsize="cohen", dv="Spiking rate per 10 ms", between="Neuron type")[["A","B", "p-tukey"]])
_, __ = format_for_annotator(out_test, "Location seed", "Lateral seed")
if len(_)>0:
    _ = [[(sub[1], sub[0]) for sub in _[0]]]
pairs.extend(_)
pvalues = np.append(pvalues, __)

out_test = pd.DataFrame(data[data["Neuron type"]=="Putative exc"]\
            .pairwise_tukey(effsize="cohen", dv="Spiking rate per 10 ms", between="Location seed")[["A","B", "p-tukey"]])
_, __ = format_for_annotator(out_test, "Neuron type", "Putative exc")

pairs.extend(_)
pvalues = np.append(pvalues, __)


out_test = pd.DataFrame(data[data["Neuron type"]=="Putative inh"]\
            .pairwise_tukey(effsize="cohen", dv="Spiking rate per 10 ms", between="Location seed")[["A","B", "p-tukey"]])
_, __ = format_for_annotator(out_test, "Neuron type", "Putative inh")

pairs.extend(_)
pvalues = np.append(pvalues, __)

if pvalues.shape[0]>0:
    annot = Annotator(ax, pairs=pairs, data=data,
                      hue=hue, y=y, x=x, hue_order=hue_order, order=order)
    (annot
     .configure(test=None, test_short_name="custom test",  text_format='star', loc='inside', verbose=0, line_height=0.05,  line_offset=20, text_offset=20)
     .set_pvalues(pvalues=pvalues)
     .set_custom_annotations([""]*len(pvalues))
     .annotate())


ttest_early_spiking = pg.ttest(data[data["Location seed"] == "Medial seed"]["Spiking rate per 10 ms"],
                  data[data["Location seed"] == "Lateral seed"]["Spiking rate per 10 ms"], paired=True)



data = pd.melt(tot_summary_late[["Lateral seed", "Medial seed", "Session id", "Neuron type"]],
               value_vars=["Lateral seed", "Medial seed"],
               id_vars=["Session id", "Neuron type"], var_name='Location seed', value_name='Spiking rate per 10 ms').groupby(
    ["Neuron type","Session id", "Location seed"]).mean().reset_index()


hue = "Location seed"
y="Spiking rate per 10 ms"
x="Neuron type"
hue_order=["Medial seed", "Lateral seed"]
order=["Putative exc", "Putative inh"]
sns.boxplot(data=data, x=x, y=y, hue=hue, showfliers=False, ax=axs[1], palette=palette_ML,
            order=order, hue_order=hue_order)
ax = sns.stripplot(data=data,  x=x, y=y, ax=axs[1],
                   hue=hue, dodge=True, size=2, color=".9", linewidth=0.1, jitter=0.1,
                   order=order, hue_order=hue_order)

handles, labels = ax.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = axs[1].legend(handles[0:2], labels[0:2])

ax.get_legend().remove()

print(pg.normality(data, dv="Spiking rate per 10 ms", group="Location seed"))
print(pg.ttest(data[data["Location seed"] == "Medial seed"]["Spiking rate per 10 ms"],
               data[data["Location seed"] == "Lateral seed"]["Spiking rate per 10 ms"])["p-val"])
print(pg.mwu(data[data["Location seed"] == "Medial seed"]["Spiking rate per 10 ms"],
             data[data["Location seed"] == "Lateral seed"]["Spiking rate per 10 ms"])["p-val"])

ttest_late_spiking = pg.ttest(data[data["Location seed"] == "Medial seed"]["Spiking rate per 10 ms"],
                  data[data["Location seed"] == "Lateral seed"]["Spiking rate per 10 ms"], paired=True)

out_test = pd.DataFrame(data[data["Location seed"]=="Medial seed"]\
            .pairwise_tukey(effsize="cohen", dv="Spiking rate per 10 ms", between="Neuron type")[["A","B", "p-tukey"]])

pairs, pvalues = format_for_annotator(out_test, "Location seed", "Medial seed")
if len(pairs)>0:
    pairs = [[(sub[1], sub[0]) for sub in pairs[0]]]

out_test = pd.DataFrame(data[data["Location seed"]=="Lateral seed"]\
            .pairwise_tukey(effsize="cohen", dv="Spiking rate per 10 ms", between="Neuron type")[["A","B", "p-tukey"]])
_, __ = format_for_annotator(out_test, "Location seed", "Lateral seed")
if len(_)>0:
    _ = [[(sub[1], sub[0]) for sub in _[0]]]
pairs.extend(_)
pvalues = np.append(pvalues, __)


out_test = pd.DataFrame(data[data["Neuron type"]=="Putative exc"]\
            .pairwise_tukey(effsize="cohen", dv="Spiking rate per 10 ms", between="Location seed")[["A","B", "p-tukey"]])
_, __ = format_for_annotator(out_test, "Neuron type", "Putative exc")

pairs.extend(_)
pvalues = np.append(pvalues, __)


out_test = pd.DataFrame(data[data["Neuron type"]=="Putative inh"]\
            .pairwise_tukey(effsize="cohen", dv="Spiking rate per 10 ms", between="Location seed")[["A","B", "p-tukey"]])
_, __ = format_for_annotator(out_test, "Neuron type", "Putative inh")

pairs.extend(_)
pvalues = np.append(pvalues, __)

if pvalues.shape[0]>0:
    annot = Annotator(ax, pairs=pairs, data=data,
                      hue=hue, y=y, x=x, hue_order=hue_order, order=order)
    (annot
     .configure(test=None, test_short_name="custom test",  text_format='star', loc='inside', verbose=0, line_height=0.05, line_offset=20, text_offset=20)
     .set_pvalues(pvalues=pvalues)
     .set_custom_annotations([""]*len(pvalues))
     .annotate())



plt.show()
