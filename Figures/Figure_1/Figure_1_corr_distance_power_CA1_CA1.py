import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import Utils.Style
from Utils.Settings import output_folder_figures_calculations

with open(f"{output_folder_figures_calculations}/temp_data_figure_1.pkl", "rb") as fp:  # Unpickling
    sessions, high_distance, low_distance, ripples_lags, ripples_lags_inverted_reference, ripples_calcs, summary_corrs = pickle.load(fp)

fig, axs = plt.subplots()
param_2 = "Distance (µm)"
#_ = summary_corrs[(summary_corrs["Comparison"]=="CA1-CA1")&(summary_corrs["Correlation"] > 0.55)]
_ = summary_corrs[(summary_corrs["Comparison"]=="CA1-CA1")]
ax = sns.regplot(data=_, x=param_2, y="Correlation",ax=axs, scatter_kws={"alpha": 0.5, "s":3, "color":(0.4,0.4,0.4)}, line_kws={"alpha": 0.8,  "color":"#D7263D", "linestyle": "--"})
ax.set_ylabel("Correlation ∫Ripple CA1-CA1")

y =_["Correlation"]
x = _[param_2]
# fit = np.polyfit(x, y, deg=1)
# predict = np.poly1d(fit)

# plt.plot( x, predict(x),color="k", alpha=0.5, linewidth=1)
# res = scipy.stats.linregress( x, y)

r, _ = pearsonr(x, y)
ax.text(.1, 0.2,f"R\u00b2={round(r**2,4)}", transform=axs.transAxes, fontsize=7, weight="bold",color="k")

plt.show()
