import xarray as xr
from tqdm import tqdm
import dill
import matplotlib.pyplot as plt
from Utils.Settings import output_folder_calculations, output_folder_figures_calculations,  var_thr
from Utils.Utils import postprocess_spike_hists, clean_ripples_calculations
from Utils.Style import palette_ML
import numpy as np
import pandas as pd

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

with open(f"{output_folder_calculations}/spike_hists_summary_per_parent_area_by_seed_location.pkl", 'rb') as f:
    spike_hists = dill.load(f)

out = []
for session_id in tqdm([q[0] for q in list(spike_hists.keys()) if q[1] == "HPF"]):
    lrs, out_hist_medial, out_hist_lateral = spike_hists[session_id, 'HPF', 'central']
    print(session_id, len(out_hist_medial) > 0, len(out_hist_lateral) > 0)
    if (len(out_hist_medial) > 0) & (len(out_hist_lateral) > 0):
        means_cut = postprocess_spike_hists(out_hist_lateral, out_hist_medial, lrs)
        means_cut_interp = means_cut.interp(assume_sorted=True, Time_ms=np.arange(-50, 130, .1),
                                            ML=np.arange(medial_lim_lm-800, lateral_lim_lm+500, 2),
                                            method="linear")#

        out.append(means_cut_interp)

summary = xr.concat(out, dim="Session")


fig, axs = plt.subplots(1)
ax = summary.mean(dim="Session").diff("Seed").plot(cmap="seismic", add_colorbar=False)
axs.invert_yaxis()

axs.vlines(x=0, ymin=axs.get_ylim()[0], ymax=axs.get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs.vlines(x=50, ymin=axs.get_ylim()[0], ymax=axs.get_ylim()[1],  colors='white', ls='--', linewidth=1)

axs.hlines(y=medial_lim_lm , xmin=axs.get_xlim()[0], xmax=axs.get_xlim()[1],  colors= palette_ML["Medial"], ls="--", linewidth=1)
axs.hlines(y=lateral_lim_lm , xmin=axs.get_xlim()[0], xmax=axs.get_xlim()[1],  colors= palette_ML["Lateral"], ls="--", linewidth=1)

plt.xlabel("Time from ripple start (ms)")
plt.ylabel("M-L (µm)")
cbar = plt.colorbar(ax, shrink=.5)
cbar.ax.set_ylabel('Δ spiking per 10 ms', rotation=270, labelpad=8)
cbar.ax.set_label("colorbar_summary") # needed to distinguish colobars

for d in ["left", "top", "bottom", "right"]:
    plt.gca().spines[d].set_visible(False)

axs.set_title("Medial seed - Lateral seed", fontsize=7)
plt.show()