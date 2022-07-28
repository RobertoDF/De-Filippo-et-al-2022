from Utils.Settings import output_folder_calculations
import dill
from Utils.Utils import format_for_annotator
import Utils.Style
from Utils.Style import palette_ML, palette_timelags
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statannotations.Annotator import Annotator
import pingouin as pg

with open(f'{output_folder_calculations}/ripples_features_sessions_across_all_ML.pkl', 'rb') as f:
    out = dill.load(f)

seed_ripples_by_hip_section_summary_common = pd.concat([q[3] for q in out])
seed_ripples_by_hip_section_summary_common = seed_ripples_by_hip_section_summary_common.reset_index().rename(columns={'index': 'Location seed'})
hue_order = ["Medial seed", "Central seed", "Lateral seed"]
order = ["Medial", "Central", "Lateral"]
hue = "Location seed"
x = "Reference"
y = "Percentage seed (%)"
data = seed_ripples_by_hip_section_summary_common
ax = sns.boxplot(data=data, hue=hue, y=y,
             x=x, palette=palette_ML, hue_order=hue_order, order=order, showfliers=False)
sns.stripplot(data=data, hue=hue, y=y,
             x=x,  hue_order=hue_order, order=order, dodge=True, size=2, color=".9", linewidth=0.6, jitter=0.1)

ax.set_ylim([0, 100])

handles, labels = ax.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = plt.legend(handles[0:3], labels[0:3])


reference = "Medial"

pg.normality(data[(data["Reference"]==reference) & (data["Location seed"]=="Lateral seed")])
pg.kruskal(data=data[data["Reference"]==reference], dv="Percentage seed (%)", between= "Location seed")


out_test = pd.DataFrame(data[data["Reference"]==reference]\
            .pairwise_tests(parametric=False, effsize="cohen", dv="Percentage seed (%)", between="Location seed")[["A","B", "p-unc"]])
pairs, pvalues = format_for_annotator(out_test, "Reference", reference)

reference = "Central"
out_test = pd.DataFrame(data[data["Reference"]==reference]\
            .pairwise_tests(parametric=False, effsize="cohen", dv="Percentage seed (%)", between="Location seed")[["A","B", "p-unc"]])
_, __ = format_for_annotator(out_test, "Reference",reference)
pairs.extend(_)
pvalues = np.append(pvalues, __)

reference = "Lateral"
out_test = pd.DataFrame(data[data["Reference"]==reference]\
            .pairwise_tests(parametric=False, effsize="cohen", dv="Percentage seed (%)", between="Location seed")[["A","B", "p-unc"]])
_, __ = format_for_annotator(out_test, "Reference", reference)
pairs.extend(_)
pvalues = np.append(pvalues, __)

if pvalues.shape[0]>0:
    annot = Annotator(ax, pairs=pairs, data=data,
                      hue=hue, y=y, x=x, palette=palette_ML, hue_order=hue_order, order=order)
    (annot
     .configure(test=None, test_short_name="custom test",  text_format='star', loc='outside', verbose=0, text_offset=-2)
     .set_pvalues(pvalues=pvalues)
     .set_custom_annotations(["*"]*len(pvalues))
     .annotate())

plt.show()
