import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.signal import hilbert
from Utils.Utils import butter_bandpass_filter
import Utils.Style
from Utils.Settings import output_folder_figures_calculations

with open(f'{output_folder_figures_calculations}/temp_data_figure_2.pkl', 'rb') as f:
    session_id, session_trajs, columns_to_keep, ripples, real_ripple_summary, lfp_per_probe, ripple_cluster_strong, \
    ripple_cluster_weak, example_session = dill.load(f)


color_palette = sns.color_palette("flare", 255)

ripple_cluster_strong.sort_values(by="M-L (µm)", inplace=True)

fs_lfp = 1250

fig = plt.figure(figsize=(5, 10))

n_rip = 2

n = 2

source = ripple_cluster_strong[ripple_cluster_strong["Start (s)"] == ripple_cluster_strong["Start (s)"].min()]


cc = []

for idx, r in ripple_cluster_strong.iterrows():

    lfp = lfp_per_probe[r["Probe number"]]

    ax = fig.add_subplot(10, 1, n)

    sample = lfp.sel(
        time=slice(source["Start (s)"].values[0] - 0.05, source["Stop (s)"].values[0] + 0.1),
        area=r["Area"])
    lowcut = 120.0
    highcut = 250.0
    filtered = butter_bandpass_filter(np.nan_to_num(sample.values), lowcut, highcut, fs_lfp, order=6)
    filtered = pd.Series(filtered, index=sample.time)
    analytic_signal = hilbert(filtered)
    amplitude_envelope = pd.Series(np.abs(analytic_signal), index=sample.time)

    plt.plot(sample.time, sample, color="k", alpha=.25)

    plt.vlines(r["Start (s)"], np.min(sample), np.max(sample), linestyles="dashed", color="k", alpha=0.9, linewidth=0.6)

    color_idx = int(r["color index"])
    # plt.text(0.05, 0.9, str(r["Probe number"]) + "-" + str(r["Area"]), color=color_palette[color_idx],
    #          transform=ax.transAxes, fontsize=6)
    n = n + 1

    bar = AnchoredSizeBar(ax.transData, 0, label='', size_vertical=0.5, loc="center right",  borderpad=1.4,
                          frameon=False)
    ax.add_artist(bar)

    plt.axis("off")
    plt.plot(filtered, color=color_palette[color_idx], alpha=.8)

    cc.append([amplitude_envelope, color_palette[color_idx]])

    if idx == ripple_cluster_strong.iloc[-1].name:
        bar = AnchoredSizeBar(ax.transData, 0.05, label="", loc="lower right", borderpad=1.4,
                              frameon=False)
        ax.add_artist(bar)

ax = fig.add_subplot(10, 1, 1)
for n in range(len(cc)):
    ax.plot([q[0] for q in cc][n], color=[q[1] for q in cc][n])
plt.axis("off")
bar = AnchoredSizeBar(ax.transData, 0, label='', size_vertical=0.1, loc="center right",  borderpad=1.4,
                      frameon=False, color="r")
ax.add_artist(bar)

fig.tight_layout()

plt.show()