from Utils.Settings import output_folder_calculations, var_thr
from Utils.Utils import clean_ripples_calculations
import dill
from Utils.Utils import corrfunc
import Utils.Style
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from Utils.Style import palette_timelags, palette_ML
from scipy.stats import pearsonr

with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)



input_rip = []
for session_id in ripples_calcs.keys():
    ripples = ripples_calcs[session_id][3].copy()
    ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)
    input_rip.append(ripples.groupby("Probe number-area").mean()["L-R (µm)"])

lr_space = pd.concat(input_rip)

medial_lim = lr_space.quantile(.33333)
lateral_lim = lr_space.quantile(.666666)
center = lr_space.median()
medial_lim_lm = medial_lim - 5691.510009765625
lateral_lim_lm = lateral_lim - 5691.510009765625

with open(f'{output_folder_calculations}/ripples_features_sessions_across_all_ML.pkl', 'rb') as f:
    out = dill.load(f)

ripples_features_summary = pd.concat([q[2] for q in out]).drop_duplicates()
ripples_features_summary = ripples_features_summary.reset_index().rename(columns={'index': 'M-L (µm)'})
ripples_features_summary.rename(columns={"Local strong":"Type"}, inplace=True)

ripples_features_summary["Type"] = ripples_features_summary["Type"].astype("category").cat.rename_categories({False:"Common ripples", True:"Strong ripples"})

g = sns.lmplot(data=ripples_features_summary, x="M-L (µm)",
                 y="Duration (s)", hue="Type", palette=palette_timelags, scatter_kws={"s": 5}, legend=False)

y = ripples_features_summary[ripples_features_summary["Type"]=="Strong ripples"]["Duration (s)"]
x = ripples_features_summary[ripples_features_summary["Type"]=="Strong ripples"]["M-L (µm)"]
r, _ = pearsonr(x, y)
ax = plt.gca()
ax.annotate(f"R\u00b2={round(r ** 2, 3)}", xy=(.01, .95), xycoords=ax.transAxes, color=palette_timelags["Strong ripples"])
r_strong = round(r ** 2, 3)

y = ripples_features_summary[ripples_features_summary["Type"]=="Common ripples"]["Duration (s)"]
x = ripples_features_summary[ripples_features_summary["Type"]=="Common ripples"]["M-L (µm)"]
r, _ = pearsonr(x, y)
ax.annotate(f"R\u00b2={round(r ** 2, 3)}", xy=(.01, .45), xycoords=ax.transAxes, color=palette_timelags["Common ripples"])
r_common = round(r ** 2, 3)
plt.legend( loc='upper right')

ymin = ax.get_ylim()[0]
ymax = ax.get_ylim()[1]
ax.vlines(x=medial_lim_lm, ymin=ymin, ymax=ymax,  colors= palette_ML["Medial"], ls="--", linewidth=1)
ax.vlines(x=lateral_lim_lm, ymin=ymin, ymax=ymax,  colors= palette_ML["Lateral"], ls="--", linewidth=1)

plt.show()