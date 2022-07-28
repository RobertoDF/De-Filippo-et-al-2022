import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import scipy
from scipy.stats import pearsonr

sns.set_theme(context='paper', style="ticks", rc={'axes.spines.right': False, 'axes.spines.top': False, "lines.linewidth": 0.5, "xtick.labelsize": 5, "ytick.labelsize": 5
                                                  , "axes.labelsize": 6, "xtick.major.size": 1, "ytick.major.size": 1 , "axes.titlesize" : 6, "legend.fontsize":6, "legend.title_fontsize":6})


with open("/temp/temp_summary_corrs.pkl", "rb") as fp:  # Unpickling
    summary_corrs = pickle.load(fp)

fig, axs = plt.subplots(figsize=(2,2))
param_2 = "Distance (µm)"
_ = summary_corrs[summary_corrs["Comparison"] == "CA1-CA1"]
ax = sns.regplot(data=_, x=param_2, y="Correlation",ax=axs, scatter_kws={"alpha": 0.5, "s":3, "color":(0.4,0.4,0.4)}, line_kws={"alpha": 0.8,  "color":"#D7263D", "linestyle": "--"})
ax.set_ylabel("Correlation ∫Ripple CA1-CA1")

y =_["Correlation"]
x = _[param_2]
# fit = np.polyfit(x, y, deg=1)
# predict = np.poly1d(fit)

# plt.plot( x, predict(x),color="k", alpha=0.5, linewidth=1)
# res = scipy.stats.linregress( x, y)

r, _ = pearsonr(x, y)
ax.text(.1,0.5,f"R\u00b2={round(r**2,4)}", transform=axs.transAxes, fontsize=7, weight="bold",color="k")

plt.show()
