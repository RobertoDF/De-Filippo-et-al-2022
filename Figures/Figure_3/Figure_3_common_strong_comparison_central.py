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

seed_strong_ripples_by_hip_section_summary = pd.concat([q[6] for q in out])
seed_strong_ripples_by_hip_section_summary = seed_strong_ripples_by_hip_section_summary.reset_index().rename(columns={'index': 'Location seed'})

seed_common_ripples_by_hip_section_summary = pd.concat([q[3] for q in out])
seed_common_ripples_by_hip_section_summary = seed_common_ripples_by_hip_section_summary.reset_index().rename(columns={'index': 'Location seed'})


seed_common_ripples_by_hip_section_summary["Type"] = "Common ripples"
seed_strong_ripples_by_hip_section_summary["Type"] = "Strong ripples"
seed_ripples_by_hip_section_summary = pd.concat([seed_common_ripples_by_hip_section_summary, seed_strong_ripples_by_hip_section_summary])

# sns.catplot(data=seed_ripples_by_hip_section_summary, hue="Type", y="Percentage seed (%)",
#              col="Reference", x="Location seed", kind="box", order=["Medial", "Central", "Lateral"],
#             col_order=["Medial", "Central", "Lateral"], palette=palette_timelags, showfliers=False)
# plt.show()

hue_order = ["Common ripples", "Strong ripples"]
order = ["Medial seed", "Central seed", "Lateral seed"]
hue = "Type"
x = "Location seed"
y = "Percentage seed (%)"

reference = "Central"
data = seed_ripples_by_hip_section_summary[seed_ripples_by_hip_section_summary["Reference"]==reference]
ax = sns.boxplot(data=data, hue=hue, y=y,
             x=x, order=order,
            hue_order=hue_order, palette=palette_timelags, showfliers=False)
ax = sns.stripplot(data=data,  hue=hue, y=y,
             x=x, hue_order=hue_order, order=order, dodge=True, size=2, color=".9", linewidth=0.6, jitter=0.1)

ax.set_ylim([0, 100])

handles, labels = ax.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = plt.legend(handles[0:2], labels[0:2])

plt.legend([],[], frameon=False)


#post-hoc tukey
location = "Medial seed"


out_test = pd.DataFrame(data[data["Location seed"]==location]\
            .pairwise_tests(parametric=False, effsize="cohen", dv="Percentage seed (%)", between="Type")[["A","B", "p-unc"]])
pairs, pvalues = format_for_annotator(out_test, "Location seed", location)

location = "Central seed"
out_test = pd.DataFrame(data[data["Location seed"]==location]\
            .pairwise_tests(parametric=False, effsize="cohen", dv="Percentage seed (%)", between="Type")[["A","B", "p-unc"]])
_, __ = format_for_annotator(out_test, "Location seed", location)
pairs.extend(_)
pvalues = np.append(pvalues, __)

location = "Lateral seed"
out_test = pd.DataFrame(data[data["Location seed"]==location]\
            .pairwise_tests(parametric=False, effsize="cohen", dv="Percentage seed (%)", between="Type")[["A","B", "p-unc"]])
_, __ = format_for_annotator(out_test, "Location seed", location)
pairs.extend(_)
pvalues = np.append(pvalues, __)

if pvalues.shape[0]>0:
    annot = Annotator(ax, pairs=pairs, data=data,
                      hue=hue, y=y, x=x, palette=palette_ML, hue_order=hue_order, order=order)
    (annot
     .configure(test=None, test_short_name="custom test",  text_format='star', loc='inside', verbose=0, text_offset=-2)
     .set_pvalues(pvalues=pvalues)
     .set_custom_annotations(["*"]*len(pvalues))
     .annotate())

plt.show()