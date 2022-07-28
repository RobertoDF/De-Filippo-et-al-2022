import seaborn as sns
import pickle
import pandas as pd
import pingouin as pg
from statannotations.Annotator import Annotator


sns.set_theme(context='paper', style="ticks", rc={'axes.spines.right': False, 'axes.spines.top': False, "lines.linewidth": 0.5, "xtick.labelsize": 5, "ytick.labelsize": 5
                                                  , "axes.labelsize": 6, "xtick.major.size": 1, "ytick.major.size": 1 , "axes.titlesize" : 6, "legend.fontsize":6, "legend.title_fontsize":6})


with open("/temp/temp_summary_corrs.pkl", "rb") as fp:  # Unpickling
    summary_corrs = pickle.load(fp)

quartiles = summary_corrs[summary_corrs["Comparison"]=="CA1-CA1"]["Correlation"].quantile([0.25, 0.5, 0.75])
quartiles_distance = summary_corrs[summary_corrs["Comparison"]=="CA1-CA1"]["Distance (µm)"].quantile([0.25, 0.5, 0.75])

_ = pd.DataFrame()
_2 = pd.DataFrame()

print(f"Quartiles:{quartiles}")
_["Correlation"] = summary_corrs[(summary_corrs["Comparison"]=="CA1-CA1")&(summary_corrs["Distance (µm)"]<quartiles_distance[0.25])]["Correlation"]
_["Distance (µm) quartiles"] = ["$\it{Q\u2081}$ distance (µm)"] * summary_corrs[(summary_corrs["Comparison"]=="CA1-CA1")&(summary_corrs["Correlation"]<quartiles[0.25])].shape[0]

_2["Correlation"] = summary_corrs[(summary_corrs["Comparison"]=="CA1-CA1")&(summary_corrs["Distance (µm)"]>quartiles_distance[0.75])]["Correlation"]
_2["Distance (µm) quartiles"] = ["$\it{Q\u2083}$ distance (µm)"] * summary_corrs[(summary_corrs["Comparison"]=="CA1-CA1")&(summary_corrs["Correlation"]>quartiles[0.75])].shape[0]

_ = _.append(_2)

print(pg.normality(_, dv="Correlation", group="Distance (µm) quartiles", method="normaltest"))
pvalues = pg.mwu(_[_["Distance (µm) quartiles"]=="$\it{Q\u2081}$ distance (µm)"]["Correlation"], _[_["Distance (µm) quartiles"]=="$\it{Q\u2083}$ distance (µm)"]["Correlation"])["p-val"]

x = "Distance (µm) quartiles"
y = "Correlation"
ax = sns.violinplot(data=_, x=x, y=y, palette=[sns.husl_palette(2)[1],sns.husl_palette(2)[0]])
#plt.setp(ax.collections, alpha=.6)
annot = Annotator(ax, [("$\it{Q\u2081}$ distance (µm)", "$\it{Q\u2083}$ distance (µm)")], data=_, x=x, y=y)
(annot
 .configure(test=None, test_short_name="custom test",  text_format='star', loc='outside', verbose=0)
 .set_pvalues(pvalues=pvalues)
 .annotate())