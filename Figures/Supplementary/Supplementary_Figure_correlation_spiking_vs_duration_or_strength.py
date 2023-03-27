import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
import dill
import pingouin as pg
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
ripples_all = ripples_all[np.isfinite(ripples_all).all(1)]

#
# r_list = []
# for session_id in ripples_all["Session"].unique():
#     x = ripples_all[ripples_all["Session"] == session_id]["∫Ripple"]
#     y = ripples_all[ripples_all["Session"] == session_id]["Number participating neurons"]
#     r, p = pearsonr(x, y)
#     r_list.append(r)
#
# r_list_strength = pd.DataFrame(r_list, columns=["r"])
# r_list_strength["Type"] = "∫Ripple - Number participating neurons"

#
# r_list = []
# for session_id in  ripples_all["Session"].unique():
#     x = ripples_all[ripples_all["Session"] == session_id]["Duration (s)"]
#     y = ripples_all[ripples_all["Session"] == session_id]["Number participating neurons"]
#     r, p = pearsonr(x, y)
#     r_list.append(r)
#
# r_list_duration = pd.DataFrame(r_list, columns=["r"])
# r_list_duration["Type"] = "Duration (s) - Number participating neurons"
# tot = pd.concat([r_list_strength, r_list_duration])
#
#
# #plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_amplitude_strength", dpi=300)
# sns.boxplot(x="Type", y="r", data=tot,
#             whis=[0, 100], width=.6)
#
# # Add in points to show each observation
# sns.stripplot(x="Type", y="r", data=tot,
#               size=4, color=".3", linewidth=0)
# plt.ylim([.6,1])
# print(pg.ttest(tot[tot["Type"]=="Duration (s) - "
#                                 "Number participating neurons"]["r"],
#                tot[tot["Type"]=="∫Ripple - Number participating neurons"]["r"])["p-val"])
#
# plt.show()


r_list = []
for session_id in ripples_all["Session"].unique():
    x = ripples_all[ripples_all["Session"] == session_id]["∫Ripple"]
    y = ripples_all[ripples_all["Session"] == session_id]["Spikes per 10 ms"]
    r, p = pearsonr(x, y)
    r_list.append(r)

r_list_strength = pd.DataFrame(r_list, columns=["r"])
r_list_strength["Type"] = "∫Ripple - Spikes per 10 ms"


r_list = []
for session_id in  ripples_all["Session"].unique():
    x = ripples_all[ripples_all["Session"] == session_id]["Duration (s)"]
    y = ripples_all[ripples_all["Session"] == session_id]["Spikes per 10 ms"]
    r, p = pearsonr(x, y)
    r_list.append(r)

r_list_duration = pd.DataFrame(r_list, columns=["r"])
r_list_duration["Type"] = "Duration (s) - Spikes per 10 ms"
tot = pd.concat([r_list_strength, r_list_duration])


sns.set_theme(context='paper', style="ticks", rc={'legend.frameon': False, "legend.fontsize": 15, "axes.labelpad": -2 ,
                                                  "legend.title_fontsize": 6, 'axes.spines.right': False, 'axes.spines.top': False,
                                                  "lines.linewidth": 0.5, "xtick.labelsize": 18, "ytick.labelsize": 18
                                                  , "xtick.major.pad": 2, "ytick.major.pad":2, "axes.labelsize": 16,
                                                  "xtick.major.size": 1.5, "ytick.major.size": 1.5, "axes.labelsize" : 20 })

f, ax = plt.subplots(figsize=(7, 10))
#plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_amplitude_strength", dpi=300)
sns.boxplot(x="Type", y="r", data=tot,
            whis=[0, 100], width=.6)

# Add in points to show each observation
sns.stripplot(x="Type", y="r", data=tot,
              size=4, color=".3", linewidth=0)

plt.xticks(rotation=8)
print(pg.ttest(tot[tot["Type"]=="Duration (s) - Spikes per 10 ms"]["r"],
               tot[tot["Type"]=="∫Ripple - Spikes per 10 ms"]["r"])["p-val"])
ax.text(.5, .8, f"*", horizontalalignment='left',
        size=30, color='black', weight='semibold', transform=ax.transAxes)

plt.xlabel("")
#plt.show()
plt.savefig(f"{output_folder_supplementary}/Figure 1-Figure supplement 3", dpi=300)
