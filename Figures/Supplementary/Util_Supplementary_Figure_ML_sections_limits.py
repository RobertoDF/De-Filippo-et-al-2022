import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dill
import seaborn as sns
from Utils.Utils import clean_ripples_calculations
from Utils.Settings import output_folder_calculations, output_folder_figures_calculations, var_thr


with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)


input_rip = []
for session_id in ripples_calcs.keys():
    ripples = ripples_calcs[session_id][3].copy()
    ripples = ripples[ripples["Area"] == "CA1"]
    ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)
    input_rip.append(ripples.groupby("Probe number-area").mean()["M-L (µm)"])

ml_space = pd.concat(input_rip)

medial_lim_ml = ml_space.quantile(.33333)
lateral_lim_ml = ml_space.quantile(.666666)

color_palette = sns.color_palette("flare", 255)

ax = sns.displot(data=ml_space, kde=True, bins=40, color="k", facecolor="w", edgecolor='black')

kdeline = ax.axes[0, 0].lines[0]
print(kdeline)
Q1 = medial_lim_ml
xs = kdeline.get_xdata()
ys = kdeline.get_ydata()
height1 = np.interp(Q1, xs, ys)

print(ax.axes[0, 0])
kdeline = ax.axes[0, 0].lines[0]
Q3 = lateral_lim_ml
xs = kdeline.get_xdata()
ys = kdeline.get_ydata()
height2 = np.interp(Q3, xs, ys)

with open(f"{output_folder_figures_calculations}/temp_for_supp_fig_ML_limits.pkl", "wb") as fp:
    dill.dump([height1, height2, xs, ys, Q1, Q3], fp)

plt.close()

