import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm
import dill
import pandas as pd
from Utils.Settings import output_folder_calculations, output_folder_supplementary, var_thr, Adapt_for_Nature_style
from Utils.Utils import Naturize
from Utils.Utils import clean_ripples_calculations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)


out = []
ripples_all = []
for session_id, sel in tqdm(ripples_calcs.items()):
    ripples = ripples_calcs[session_id][3].copy()
    sel_probe = ripples_calcs[session_id][5].copy()
    ripples = ripples[ripples["Probe number"] == sel_probe]
    ripples["Session"] = session_id
    ripples_all.append(ripples[["Duration (s)", "Amplitude (mV)", "Session"]])

ripples_all = pd.concat(ripples_all)
with sns.plotting_context("paper", font_scale=2.0):
    g = sns.lmplot(data=ripples_all, col="Session", y="Amplitude (mV)", x="Duration (s)", col_wrap=7,
                   scatter_kws={"alpha": 0.6, "s": 5, "color": (0.4, 0.4, 0.4)},
                   line_kws={"alpha": 0.8, "color": "#D7263D", "linestyle": "--"})

r_list = []
x_y = []

for ax, session_id in zip(g.axes.flat, ripples_all["Session"].unique()):
    y = ripples_all[ripples_all["Session"] == session_id]["Amplitude (mV)"]
    x = ripples_all[ripples_all["Session"] == session_id]["Duration (s)"]
    r, p = pearsonr(x, y)
    r_list.append(r)
    x_y.append((np.mean(x),np.mean(y)))

    # ax.text(.1, .2, f"R\u00b2={round(r ** 2, 2)}", horizontalalignment='left',
    #         size='xx-large', color='black', weight='semibold')
    ax.text(.1, .8, f"r={round(r, 2)}", horizontalalignment='left',
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



if Adapt_for_Nature_style is True:
    Naturize()

plt.show()
#plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_duration_amplitude", dpi=300)
r_list = pd.DataFrame(r_list, columns=["r"])
print(r_list.mean(), "±", r_list.sem())

fig, ax2 = plt.subplots( figsize=(10, 5))
sns.boxplot( x="r", data=r_list,
            whis=[0, 100], width=.6, ax=ax2)

# Add in points to show each observation
sns.stripplot( x="r", data=r_list,
              size=4, color=".3", linewidth=0, ax=ax2)
ax.set_xlim([0,1])
ax.set(xlabel="r")
plt.show()