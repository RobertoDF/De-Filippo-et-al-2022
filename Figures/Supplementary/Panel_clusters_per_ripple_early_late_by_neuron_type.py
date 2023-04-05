import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from Utils.Style import palette_ML
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

fig, axs = plt.subplots(1,2,figsize=(16,8))
data=summary_fraction_active_clusters_per_ripples_early_by_neuron_type.groupby(["Neuron type", "Session id", "Location seed"]).mean().reset_index()
data["Fraction active neurons per ripple (%)"] = data["Fraction active neurons per ripple (%)"] * 100


PROPS = {
    'medianprops':{'color':'white'},
}

hue = "Location seed"
y = "Fraction active neurons per ripple (%)"
x = "Neuron type"
hue_order = ["Medial seed", "Lateral seed"]
order = ["Putative exc", "Putative inh"]
ax = sns.boxplot(data=data, x=x, y=y, palette=palette_ML, hue=hue,  ax=axs[0],showfliers=False, order=order, hue_order=hue_order, **PROPS)
ax.get_legend().remove()

ax = sns.stripplot(data=data, x=x, y=y, hue=hue, ax=axs[0],  dodge=True, size=2,
                   color=".9", linewidth=0.1, jitter=0.1, order=order, hue_order=hue_order)

ax.get_legend().remove()

print(pg.normality(data, dv="Fraction active neurons per ripple (%)", group="Location seed"))
print(pg.ttest(data[data["Location seed"]=="Medial seed"]["Fraction active neurons per ripple (%)"],
               data[data["Location seed"]=="Lateral seed"]["Fraction active neurons per ripple (%)"], paired=True)["p-val"])
print(pg.mwu(data[data["Location seed"]=="Medial seed"]["Fraction active neurons per ripple (%)"],
             data[data["Location seed"]=="Lateral seed"]["Fraction active neurons per ripple (%)"])["p-val"])


out_test = pd.DataFrame(data[data["Location seed"]=="Medial seed"]\
            .pairwise_tukey(effsize="cohen", dv="Fraction active neurons per ripple (%)", between="Neuron type")[["A","B", "p-tukey"]])

pairs, pvalues = format_for_annotator(out_test, "Location seed", "Medial seed")
if len(pairs)>0:
    pairs = [[(sub[1], sub[0]) for sub in pairs[0]]]

out_test = pd.DataFrame(data[data["Location seed"]=="Lateral seed"]\
            .pairwise_tukey(effsize="cohen", dv="Fraction active neurons per ripple (%)", between="Neuron type")[["A","B", "p-tukey"]])
_, __ = format_for_annotator(out_test, "Location seed", "Lateral seed")
if len(_)>0:
    _ = [[(sub[1], sub[0]) for sub in _[0]]]
pairs.extend(_)
pvalues = np.append(pvalues, __)

out_test = pd.DataFrame(data[data["Neuron type"]=="Putative exc"]\
            .pairwise_tukey(effsize="cohen", dv="Fraction active neurons per ripple (%)", between="Location seed")[["A","B", "p-tukey"]])
_, __ = format_for_annotator(out_test, "Neuron type", "Putative exc")

pairs.extend(_)
pvalues = np.append(pvalues, __)


out_test = pd.DataFrame(data[data["Neuron type"]=="Putative inh"]\
            .pairwise_tukey(effsize="cohen", dv="Fraction active neurons per ripple (%)", between="Location seed")[["A","B", "p-tukey"]])
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


data = summary_fraction_active_clusters_per_ripples_late_by_neuron_type.groupby(["Neuron type", "Session id", "Location seed"]).mean().reset_index()
data["Fraction active neurons per ripple (%)"] = data["Fraction active neurons per ripple (%)"]  * 100


hue = "Location seed"
y = "Fraction active neurons per ripple (%)"
x = "Neuron type"
hue_order = ["Medial seed", "Lateral seed"]
order = ["Putative exc", "Putative inh"]
sns.boxplot(data=data, x=x, y=y, hue=hue, palette=palette_ML,  ax=axs[1],showfliers=False, order=order, hue_order=hue_order)
ax = sns.stripplot(data=data, x=x, y=y, hue=hue,  ax=axs[1],  dodge=True, size=2,
                   color=".9", linewidth=0.1, jitter=0.1, order=order, hue_order=hue_order)


print(pg.normality(data, dv="Fraction active neurons per ripple (%)", group="Location seed"))
print(pg.ttest(data[data["Location seed"] == "Medial seed"]["Fraction active neurons per ripple (%)"],
               data[data["Location seed"] == "Lateral seed"]["Fraction active neurons per ripple (%)"], paired=True)["p-val"])
print(pg.mwu(data[data["Location seed"] == "Medial seed"]["Fraction active neurons per ripple (%)"],
             data[data["Location seed"] == "Lateral seed"]["Fraction active neurons per ripple (%)"])["p-val"])

handles, labels = ax.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = axs[1].legend(handles[0:2], labels[0:2])

out_test = pd.DataFrame(data[data["Location seed"]=="Medial seed"]\
            .pairwise_tukey(effsize="cohen", dv="Fraction active neurons per ripple (%)", between="Neuron type")[["A","B", "p-tukey"]])

pairs, pvalues = format_for_annotator(out_test, "Location seed", "Medial seed")
if len(pairs)>0:
    pairs = [[(sub[1], sub[0]) for sub in pairs[0]]]

out_test = pd.DataFrame(data[data["Location seed"]=="Lateral seed"]\
            .pairwise_tukey(effsize="cohen", dv="Fraction active neurons per ripple (%)", between="Neuron type")[["A","B", "p-tukey"]])
_, __ = format_for_annotator(out_test, "Location seed", "Lateral seed")
if len(_)>0:
    _ = [[(sub[1], sub[0]) for sub in _[0]]]

pairs.extend(_)
pvalues = np.append(pvalues, __)

out_test = pd.DataFrame(data[data["Neuron type"]=="Putative exc"]\
            .pairwise_tukey(effsize="cohen", dv="Fraction active neurons per ripple (%)", between="Location seed")[["A","B", "p-tukey"]])
_, __ = format_for_annotator(out_test, "Neuron type", "Putative exc")

pairs.extend(_)
pvalues = np.append(pvalues, __)


out_test = pd.DataFrame(data[data["Neuron type"]=="Putative inh"]\
            .pairwise_tukey(effsize="cohen", dv="Fraction active neurons per ripple (%)", between="Location seed")[["A","B", "p-tukey"]])
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
plt.show()
