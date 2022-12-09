from Utils.Settings import output_folder_supplementary , output_folder_calculations, Adapt_for_Nature_style
from Utils.Utils import Naturize
import dill
import pingouin as pg
import Utils.Style
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pylustrator
from Utils.Utils import format_for_annotator
from Utils.Style import palette_ML
from statannotations.Annotator import Annotator

pylustrator.start()

with open(f'{output_folder_calculations}/ripples_features_sessions_across_all_ML.pkl', 'rb') as f:
    out = dill.load(f)

ripples_features_summary_strong = pd.concat([q[1] for q in out]).drop_duplicates()
ripples_features_summary_strong = ripples_features_summary_strong[ripples_features_summary_strong["Reference"]!="Central"]
ripples_features_summary_common = pd.concat([q[7] for q in out]).drop_duplicates()
ripples_features_summary_common = ripples_features_summary_common[ripples_features_summary_common["Reference"]!="Central"]


fig, axs = plt.subplots(1, 2, figsize=(8, 4))
g = sns.boxplot(ax=axs[0], data=ripples_features_summary_strong, showfliers=False, x="Reference",
                 y="Strength conservation index", palette=palette_ML, order=["Medial",  "Lateral"])

g = sns.stripplot(ax=axs[0], data=ripples_features_summary_strong, x="Reference",
                 y="Strength conservation index",dodge=True, size=2, color=".9", linewidth=0.6, jitter=0.1, order=["Medial",  "Lateral"])

ripples_features_summary_strong = ripples_features_summary_strong.infer_objects()
print(pg.normality(ripples_features_summary_strong, dv="Strength conservation index", group="Reference"))

out_test = pd.DataFrame(ripples_features_summary_strong\
            .pairwise_tukey(effsize="cohen", dv="Strength conservation index", between="Reference")[["A", "B", "p-tukey", "cohen"]])
out_test = out_test[out_test["p-tukey"]<0.05]
pairs = list(out_test[["A", "B"]].apply(tuple, axis=1))
pvalues = out_test["p-tukey"].values

if pvalues.shape[0]>0:
    annot = Annotator(g, pairs=pairs, data=ripples_features_summary_strong,
                      x="Reference",
                      y="Strength conservation index", order=["Medial",  "Lateral"])
    (annot
     .configure(test=None, test_short_name="custom test",  text_format='star', loc='inside', verbose=0)
     .set_pvalues(pvalues=pvalues)
     .set_custom_annotations(["*"]*len(pvalues))
     .annotate())
    axs[0].text(.5, .7, f"Cohen's d: {abs(round(out_test['cohen'].values[0], 2))}",
           transform=axs[0].transAxes,
           fontsize=6, ha='center', va='center');


axs[0].set_title("Strong ripples")


g = sns.boxplot(ax=axs[1], data=ripples_features_summary_common, showfliers=False, x="Reference",
                 y="Strength conservation index", palette=palette_ML, order=["Medial",  "Lateral"])

g = sns.stripplot(ax=axs[1], data=ripples_features_summary_common, x="Reference",
                 y="Strength conservation index",dodge=True, size=2, color=".9", linewidth=0.6, jitter=0.1, order=["Medial",  "Lateral"])
axs[1].set_title("Common ripples")
ripples_features_summary_common = ripples_features_summary_common.infer_objects()
print(pg.normality(ripples_features_summary_common, dv="Strength conservation index", group="Reference"))

out_test = pd.DataFrame(ripples_features_summary_common\
            .pairwise_tukey(effsize="cohen", dv="Strength conservation index", between="Reference")[["A", "B", "p-tukey", "cohen"]])
out_test = out_test[out_test["p-tukey"]<0.05]
pairs = list(out_test[["A", "B"]].apply(tuple, axis=1))
pvalues = out_test["p-tukey"].values



if pvalues.shape[0]>0:
    annot = Annotator(g, pairs=pairs, data=ripples_features_summary_common,
                      x="Reference",
                      y="Strength conservation index", order=["Medial",  "Lateral"])
    (annot
     .configure(test=None, test_short_name="custom test",  text_format='star', loc='inside', verbose=0)
     .set_pvalues(pvalues=pvalues)
     .set_custom_annotations(["*"]*len(pvalues))
     .annotate())
    axs[1].text(.5, .7, f"Cohen's d: {abs(round(out_test['cohen'].values[0], 2))}",
                transform=axs[1].transAxes,
                fontsize=6, ha='center', va='center');

###
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.027500, 0.955000])
plt.figure(1).texts[0].set_text("A")
plt.figure(1).texts[0].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.490000, 0.955000])
plt.figure(1).texts[1].set_text("B")
plt.figure(1).texts[1].set_weight("bold")

if Adapt_for_Nature_style is True:
    Naturize()

#% end: automatic generated code from pylustrator
#plt.show()
plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_7", dpi=300)