import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import hilbert
from Utils.Utils import butter_bandpass_filter,  get_ML_limits
from itertools import chain
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import scipy
import Utils.Style
from Utils.Settings import output_folder_figures_calculations, var_thr

ml_space = get_ML_limits(var_thr)
ml_space = ml_space + 5691.510009765625

with open(f"{output_folder_figures_calculations}/temp_data_figure_4.pkl", 'rb') as f:
    space_sub_spike_times, target_area, units, field_to_use_to_compare, \
    session_id_example, lfp, lfp_per_probe, \
    ripple_cluster_lateral_seed, ripple_cluster_medial_seed, source_area, ripples, \
    tot_summary_early, summary_fraction_active_clusters_per_ripples_early, \
    summary_fraction_active_clusters_per_ripples_early_by_neuron_type, \
    tot_summary_late, summary_fraction_active_clusters_per_ripples_late, \
    summary_fraction_active_clusters_per_ripples_late_by_neuron_type, \
    tot_summary, summary_fraction_active_clusters_per_ripples, \
    summary_fraction_active_clusters_per_ripples_by_neuron_type = dill.load(f)


window = (0.1, 0.2)
time_center = ripple_cluster_lateral_seed.iloc[0]["Start (s)"]
time_space_sub_spike_times = {
    cluster_id: spikes[(spikes > time_center - window[0]) & (spikes < time_center + window[1])] for cluster_id, spikes
    in space_sub_spike_times.items()}


probe_ids = units[units[field_to_use_to_compare] == target_area].sort_values("left_right_ccf_coordinate")[
    "probe_id"].unique()

probes_list = []

for probe_id in probe_ids:
    lr = units[units[field_to_use_to_compare] == target_area].groupby("probe_id").mean().loc[probe_id][
        "left_right_ccf_coordinate"]
    temp = {cluster_id: spikes for cluster_id, spikes in time_space_sub_spike_times.items() if
            units.loc[cluster_id, :]["probe_id"] == probe_id}
    temp = {i: j for i, j in temp.items() if j.size > 0}  # delete empties

    probes_list.append([lr, temp])

out_hist = []
for _ in probes_list:
    y = np.histogram(np.array(list(chain(*_[1].values()))) - time_center, bins=np.arange(-.1, .2, .01))[0]

    upsamped = scipy.interpolate.CubicSpline(np.arange(-.1 + .005, .2 - .005, .01), y)

    out_hist.append(upsamped(np.arange(-.1 + .0005, .2 - .0005, .001)))


color_palette = sns.color_palette("flare", 255)

fig, axs = plt.subplots(len(probes_list) + 1, figsize=(8, 8))
for n, _ in enumerate(probes_list):
    axs[n + 1].eventplot(np.array(list(_[1].values())) - time_center, linewidths=.5, linelengths=2,
                         color=color_palette[round(((_[0] - ml_space.min()) / (ml_space.max() - ml_space.min())) * 255)]);
    ylims = axs[n + 1].get_ylim()
    axs[n + 1].plot(np.arange(-.1 + .0005, .2 - .0005, .001), out_hist[n] / max(out_hist[n])*(axs[n+1].get_ylim()[1]-axs[n+1].get_ylim()[1]/50),
                    color=color_palette[round(((_[0] - ml_space.min()) / (ml_space.max() - ml_space.min())) * 255)], linewidth=1, alpha=.6)
    axs[n + 1].vlines([np.arange(0, 0.15, 0.025)], axs[n + 1].get_ylim()[0], axs[n + 1].get_ylim()[1], linestyle="--",
                      color="k", alpha=.5)
    axs[n + 1].set_xlim((-.1, .2))
    axs[n + 1].set_ylim(ylims)

    if n + 1 == len(probes_list):

        axs[n + 1].axis("on")
        axs[n + 1].axes.get_yaxis().set_visible(False)
        axs[n + 1].set_frame_on(False)
        xmin, xmax = axs[n + 1].get_xaxis().get_view_interval()
        ymin, ymax = axs[n + 1].get_yaxis().get_view_interval()
        axs[n + 1].add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=3))
        axs[n + 1].set_xlabel("Time from ripple start (ms)")
    else:
        axs[n + 1].axis("off")

fs_lfp = 1250
lfp = lfp_per_probe[len(lfp_per_probe) - 1]

sample = lfp.sel(
    time=slice(time_center - 0.1, time_center + 0.2),
    area="CA1")

lowcut = 120.0
highcut = 250.0
filtered = butter_bandpass_filter(np.nan_to_num(sample.values), lowcut, highcut, fs_lfp, order=6)
filtered = pd.Series(filtered, index=sample.time)
analytic_signal = hilbert(filtered)
amplitude_envelope = pd.Series(np.abs(analytic_signal), index=sample.time)


axs[0].plot(sample.time, sample, color="k", alpha=.5)
bar = AnchoredSizeBar(axs[0].transData, 0, label='', size_vertical=0.5, loc="upper right",  borderpad=1.4,
                          frameon=False)
axs[0].add_artist(bar)
axs[0].axis("off")
axs[0].plot(sample.time, filtered, alpha=.8,
            color=color_palette[round(((probes_list[len(lfp_per_probe) - 1][0] - ml_space.min()) / (ml_space.max() - ml_space.min())) * 255)])
axs[0].set_xlim((time_center - 0.1, time_center + 0.2))
axs[0].vlines([np.arange(0, 0.15, 0.025) + time_center], axs[0].get_ylim()[0], axs[0].get_ylim()[1], linestyle="--",
              color="k", alpha=.5)


def update_ticks(x, pos):
    return int(x * 1000)

axs[n + 1].xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))

plt.show()