import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm
import dill
from Utils.Settings import output_folder_calculations, output_folder_supplementary, var_thr, Adapt_for_Nature_style
from Utils.Utils import Naturize
from Utils.Utils import clean_ripples_calculations
import pandas as pd
import matplotlib.pyplot as plt


with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)


out = []
ripples_all = []
for session_id, sel in tqdm(ripples_calcs.items()):
    ripples = ripples_calcs[session_id][3].copy()
    sel_probe = ripples_calcs[session_id][5].copy()
    ripples = ripples[ripples["Probe number"] == sel_probe]
    ripples["Session"] = session_id
    ripples["Spikes per 10 ms"] = ripples["Number spikes"]/(ripples["Duration (s)"]*100)
    ripples_all.append(ripples[["Spikes per 10 ms", "Number spikes", "Number participating neurons", "∫Ripple", "Duration (s)", "Session"]])

ripples_all = pd.concat(ripples_all)
with sns.plotting_context("paper", font_scale=2.0):
    g = sns.lmplot(data=ripples_all, col="Session", x="Spikes per 10 ms", y="∫Ripple", col_wrap=7,
                   scatter_kws={"alpha": 0.6, "s": 5, "color": (0.4, 0.4, 0.4)},
                   line_kws={"alpha": 0.8, "color": "#D7263D", "linestyle": "--"})

for ax, session_id in zip(g.axes.flat, ripples_all["Session"].unique()):
    x = ripples_all[ripples_all["Session"] == session_id]["Spikes per 10 ms"]
    y = ripples_all[ripples_all["Session"] == session_id]["∫Ripple"]
    r, p = pearsonr(x, y)

    # ax.text(.1, .2, f"R\u00b2={round(r ** 2, 2)}", horizontalalignment='left',
    #         size='xx-large', color='black', weight='semibold')
    ax.text(.1, .8, f"r={round(r , 2)}", horizontalalignment='left',
            size='xx-large', color='black', weight='semibold', transform = ax.transAxes)
    if p < 0.0005:
        ax.text(.4, .8, f"***", horizontalalignment='left',
                size='xx-large', color='black', weight='semibold', transform = ax.transAxes)
    elif p < 0.005:
        ax.text(.4, .8, f"**", horizontalalignment='left',
                size='xx-large', color='black', weight='semibold', transform = ax.transAxes)
    elif p < 0.05:
        ax.text(.4, .8, f"*", horizontalalignment='left',
                size='xx-large', color='black', weight='semibold', transform = ax.transAxes)

plt.show()
#plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_amplitude_strength", dpi=300)
