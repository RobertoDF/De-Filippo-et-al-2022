import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm
import dill
from Utils.Settings import output_folder_calculations, output_folder_supplementary, var_thr
from Utils.Utils import clean_ripples_calculations
import pandas as pd
import matplotlib.pyplot as plt


with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)


out = []
ripples_all = []
for session_id, sel in tqdm(ripples_calcs.items()):
    ripples = ripples_calcs[session_id][3].copy()
    ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)
    ripples["Session"] = session_id
    ripples_all.append(ripples[["Duration (s)", "∫Ripple", "Session"]])

ripples_all = pd.concat(ripples_all)
with sns.plotting_context("paper", font_scale=2.0):
    g = sns.lmplot(data=ripples_all, col="Session", x="∫Ripple", y="Duration (s)", col_wrap=7,
                   scatter_kws={"alpha": 0.6, "s": 5, "color": (0.4, 0.4, 0.4)},
                   line_kws={"alpha": 0.8, "color": "#D7263D", "linestyle": "--"})

for ax, session_id in zip(g.axes.flat, ripples_all["Session"].unique()):
    y = ripples_all[ripples_all["Session"] == session_id]["∫Ripple"]
    x = ripples_all[ripples_all["Session"] == session_id]["Duration (s)"]
    r, _ = pearsonr(x, y)

    ax.text(.1, .2, f"R\u00b2={round(r ** 2, 2)}", horizontalalignment='left',
            size='xx-large', color='black', weight='semibold')

#plt.show()
plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_duration_strength", dpi=300)
