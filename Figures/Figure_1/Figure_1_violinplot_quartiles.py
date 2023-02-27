import seaborn as sns
import pickle
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
import Utils.Style
from Utils.Settings import output_folder_figures_calculations

with open(f"{output_folder_figures_calculations}/temp_data_figure_1.pkl", "rb") as fp:  # Unpickling
    sessions, high_distance, low_distance, ripples_lags, ripples_lags_inverted_reference, ripples_calcs, summary_corrs, distance_tabs = pickle.load(fp)

quartiles = summary_corrs[summary_corrs["Comparison"]=="CA1-CA1"]["Correlation"].quantile([0.25, 0.5, 0.75])
quartiles_distance = summary_corrs[summary_corrs["Comparison"]=="CA1-CA1"]["Distance (µm)"].quantile([0.25, 0.5, 0.75])

_ = pd.DataFrame()
_2 = pd.DataFrame()


_["Distance (µm)"] = summary_corrs[(summary_corrs["Comparison"]=="CA1-CA1")&(summary_corrs["Correlation"]<quartiles[0.25])]["Distance (µm)"]
_["Correlation quartiles"] = ["Q₁ correlation"] * summary_corrs[(summary_corrs["Comparison"]=="CA1-CA1")&(summary_corrs["Correlation"]<quartiles[0.25])].shape[0]

_2["Distance (µm)"] = summary_corrs[(summary_corrs["Comparison"]=="CA1-CA1")&(summary_corrs["Correlation"]>quartiles[0.75])]["Distance (µm)"]
_2["Correlation quartiles"] = ["Q₄ correlation"] * summary_corrs[(summary_corrs["Comparison"]=="CA1-CA1")&(summary_corrs["Correlation"]>quartiles[0.75])].shape[0]

violinplot_data = _.append(_2)

# print(pg.normality(_, dv="Distance (µm)", group="Correlation quartiles", method="normaltest"))
pvalues = pg.mwu(violinplot_data[violinplot_data["Correlation quartiles"] == "Q₁ correlation"]["Distance (µm)"], violinplot_data[violinplot_data["Correlation quartiles"]=="Q₄ correlation"]["Distance (µm)"])["p-val"]

x = "Correlation quartiles"
y = "Distance (µm)"
ax = sns.violinplot(data=violinplot_data, x=x, y=y, palette="husl")

annot = Annotator(ax, [("Q₁ correlation", "Q₄ correlation")], data=violinplot_data, x=x, y=y)
(annot
 .configure(test=None, test_short_name="custom test",  text_format='star', loc='outside', verbose=0)
 .set_pvalues(pvalues=pvalues)
 .set_custom_annotations(["*"])
 .annotate())

ax.set_ylim([0, None])
plt.show()