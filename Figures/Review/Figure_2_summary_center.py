import dill
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from Utils.Utils import plot_trajs_interp, interpolate_and_reindex
import Utils.Style
from Utils.Style import palette_timelags
from Utils.Settings import output_folder_calculations


with open(f'{output_folder_calculations}/trajectories_by_duration.pkl', 'rb') as f:
    trajectories_by_strength = dill.load(f)

fig, axs = plt.subplots(1, 3, figsize=(12, 8))

plot_trajs_interp(trajectories_by_strength[trajectories_by_strength["Location"] == "Center"], axs, [-15, 30])

interp_trajs = trajectories_by_strength[trajectories_by_strength["Location"] == "Center"]\
    .groupby(["Session", "Type"])\
    .progress_apply(lambda group: interpolate_and_reindex(group.set_index("M-L (µm)")["Lag (ms)"]))
interp_trajs = pd.DataFrame(interp_trajs).reset_index()
interp_trajs.columns = ["Session", "Type", "M-L (µm)", "Lag (ms)"]

interp_trajs = interp_trajs[interp_trajs["M-L (µm)"].between (7250 - 5691.510009765625, 9250 - 5691.510009765625)]

interp_trajs = interp_trajs[interp_trajs["Type"] != "Total ripples"]

sns.lineplot(data=interp_trajs.groupby(["Type", "M-L (µm)"]).mean().reset_index(),
             y="Lag (ms)", x="M-L (µm)", hue=interp_trajs.groupby(["Type", "M-L (µm)"]).mean().reset_index()["Type"],
             palette=palette_timelags, ax=axs[2])

y = interp_trajs[interp_trajs["Type"] == "Strong ripples"].groupby("M-L (µm)").mean().reset_index()["Lag (ms)"]
x = interp_trajs[interp_trajs["Type"] == "Strong ripples"].groupby("M-L (µm)").mean().reset_index()["M-L (µm)"]
error = interp_trajs[interp_trajs["Type"] == "Strong ripples"].groupby("M-L (µm)").std().reset_index()["Lag (ms)"] / \
        np.sqrt(interp_trajs["Session"].unique().shape[0])
axs[2].fill_between(x, y-error, y+error, alpha=.3, color=palette_timelags["Strong ripples"])

y = interp_trajs[interp_trajs["Type"] == "Common ripples"].groupby("M-L (µm)").mean().reset_index()["Lag (ms)"]
x = interp_trajs[interp_trajs["Type"] == "Common ripples"].groupby("M-L (µm)").mean().reset_index()["M-L (µm)"]
error = interp_trajs[interp_trajs["Type"] == "Strong ripples"].groupby("M-L (µm)").std().reset_index()["Lag (ms)"] / \
        np.sqrt(interp_trajs["Session"].unique().shape[0])
axs[2].fill_between(x, y-error, y+error, alpha=.3, color=palette_timelags["Common ripples"])

# y = interp_trajs[interp_trajs["Type"]=="Total ripples"].groupby("M-L (µm)").mean().reset_index()["Lag (ms)"]
# x = interp_trajs[interp_trajs["Type"]=="Total ripples"].groupby("M-L (µm)").mean().reset_index()["M-L (µm)"]
# error = interp_trajs[interp_trajs["Type"]=="Total ripples"].groupby("M-L (µm)").std().reset_index()["Lag (ms)"] / \
#         np.sqrt(interp_trajs["Session"].unique().shape[0])
# axs[2].fill_between(x, y-error, y+error, alpha=.3, color=palette_timelags["Total ripples"])

axs[2].set_ylim([-10, 20])
plt.legend(loc='upper left', title="")

plt.show()