import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from matplotlib.colors import rgb2hex
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.signal import hilbert
from Utils.Utils import butter_bandpass_filter, clean_ripples_calculations
import Utils.Style
from Utils.Settings import output_folder_figures_calculations

with open(f"{output_folder_figures_calculations}/temp_data_supp_figure_1.pkl", "rb") as fp:  # Unpickling
    data_area, ripple_power, session_id, session_summary, summary_table, ripples = pickle.load(fp)

print(session_id)

with open(f'/alzheimer/Roberto/Allen_Institute/Processed_lfps/lfp_per_probe_{session_id}.pkl', 'rb') as f:
    lfp_per_probe = dill.load(f)

ripple_sel_start = 5213.10845

ripple_selected = ripples.loc[ripples['Start (s)'].sub(ripple_sel_start).abs().idxmin(), :]

fs_lfp = 1250

fig = plt.figure(figsize=(15, 22.5))

n = 1
colors = iter(plt.cm.Dark2(np.linspace(0, 1, 8)))

for q, lfp in enumerate(lfp_per_probe):

    color = rgb2hex(next(colors))
    if "CA1" in lfp.area:
        ax = fig.add_subplot(10, 3, n)
        sample = lfp.sel(
            time=slice(ripple_selected["Start (s)"] - 0.05, ripple_selected["Stop (s)"] + 0.2),
            area="CA1")
        lowcut = 120.0
        highcut = 250.0
        filtered = butter_bandpass_filter(np.nan_to_num(sample.values), lowcut, highcut, fs_lfp, order=6)
        analytic_signal = hilbert(filtered)
        amplitude_envelope = pd.Series(np.abs(analytic_signal), index=sample.time)

        plt.plot(sample, color=color, alpha=.8)
        plt.text(0.05, 0.95, "CA1", color=color, transform=ax.transAxes, fontsize=6)
        n = n + 1

        bar = AnchoredSizeBar(ax.transData, 0, label='', size_vertical=0.25, loc="lower right", borderpad=1.5,
                              frameon=False)
        ax.add_artist(bar)
        plt.axis("off")
        plt.plot(filtered, color="k", alpha=.8)
        ax = fig.add_subplot(10, 3, n)
        n = n + 1
        plt.plot(amplitude_envelope, color="k", alpha=.8)
        plt.fill_between(amplitude_envelope[ripple_selected["Start (s)"] - 0.05: ripple_selected ["Stop (s)"] + 0.05].index,
                amplitude_envelope[ripple_selected["Start (s)"] - 0.05: ripple_selected["Stop (s)"] + 0.05],
                         color="#FE4A49")
        ax.set_ylim([0, 0.6])
        if q == 0:
            plt.title("∫Ripple", color="#FE4A49")
        plt.axis("off")
        ax = fig.add_subplot(10, 3, n)
        n = n + 1
        plt.plot(sample.time, sample,  color="k", alpha=.8)
        plt.fill_between(sample.sel(time=slice(ripple_selected["Start (s)"], ripple_selected["Stop (s)"]  + 0.2)).time,
                         sample.sel(time=slice(ripple_selected["Start (s)"], ripple_selected["Stop (s)"]  + 0.2)), color="#86A2BA")
        ax.set_ylim([-2, 2])
        if q ==0:
            plt.title("RIVD", color="#86A2BA")
        plt.axis("off")


        if "DG" in lfp.area:
            ax = fig.add_subplot(10, 3, n)
            sample = lfp.sel(
                time=slice(ripple_selected["Start (s)"] - 0.05, ripple_selected["Stop (s)"]  + 0.2),
                area="DG")
            lowcut = 120.0
            highcut = 250.0
            filtered = butter_bandpass_filter(np.nan_to_num(sample.values), lowcut, highcut, fs_lfp, order=6)
            analytic_signal = hilbert(filtered)
            amplitude_envelope = pd.Series(np.abs(analytic_signal), index=sample.time)

            plt.plot(sample, color=color, alpha=.8)
            plt.text(0.05, 0.95, "DG", color=color, transform=ax.transAxes, fontsize=6)
            n = n + 1
            if q == len(lfp_per_probe) - 1:
                bar = AnchoredSizeBar(ax.transData, fs_lfp / 20, label="", loc="lower right", borderpad=1.5,
                                      frameon=False)
                ax.add_artist(bar)
            bar = AnchoredSizeBar(ax.transData, 0, label='', size_vertical=0.25, loc="lower right", borderpad=1.5,
                                  frameon=False)
            ax.add_artist(bar)
            plt.axis("off")

            plt.plot(filtered, color="k", alpha=.8)
            ax = fig.add_subplot(10, 3, n)
            n = n + 1
            plt.plot(amplitude_envelope, color="k", alpha=.8)
            plt.fill_between(
                amplitude_envelope[ripple_selected["Start (s)"] - 0.05: ripple_selected["Stop (s)"] + 0.05].index,
                amplitude_envelope[ripple_selected["Start (s)"] - 0.05: ripple_selected["Stop (s)"] + 0.05],
                color="#FE4A49")

            ax.set_ylim([0, 0.6])
            plt.axis("off")
            ax = fig.add_subplot(10, 3, n)
            n = n + 1
            plt.plot(sample.time, sample, color="k", alpha=.8)
            plt.fill_between(sample.sel(
                time=slice(ripple_selected["Start (s)"], ripple_selected["Stop (s)"] + 0.2)).time,
                             sample.sel(time=slice(ripple_selected["Start (s)"],
                                                   ripple_selected["Stop (s)"] + 0.2)), color="#86A2BA")
            ax.set_ylim([-2, 2])

            plt.axis("off")
fig.tight_layout()

plt.show()

