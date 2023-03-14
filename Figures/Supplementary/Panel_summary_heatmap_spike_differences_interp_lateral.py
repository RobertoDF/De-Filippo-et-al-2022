import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from Utils.Settings import output_folder_calculations
from Utils.Utils import postprocess_spike_hists_strength
from tqdm import tqdm

with open(f"{output_folder_calculations}/spike_hists_summary_per_parent_area_by_seed_location_and_strength.pkl", 'rb') as f:
    spike_hists = dill.load(f)

out = []
for session_id in tqdm([q[0] for q in list(spike_hists.keys()) if q[1] == "HPF"]):
    lrs, out_hist_strong, out_hist_common = spike_hists[session_id, 'HPF', 'lateral']
    print(session_id, len(out_hist_strong) > 0, len(out_hist_common) > 0)
    if (len(out_hist_strong) > 5) & (len(out_hist_common) > 5):
        means_cut = postprocess_spike_hists_strength(out_hist_common, out_hist_strong, lrs)
        means_cut_interp = means_cut.interp(assume_sorted=True, Time_ms=np.arange(-50, 130, .1),
                                            ML=np.arange(1174, 3745, 2),
                                            method="linear")#

        out.append(means_cut_interp)

summary = xr.concat(out, dim="Session")


fig, axs = plt.subplots(1, figsize=(5,5))
ax = summary.mean(dim="Session").diff("Strength").plot(cmap="seismic", add_colorbar=False, vmin=-20, vmax=20)
axs.invert_yaxis()
axs.set_ylim(3600, 1250)
axs.vlines(x=0, ymin=axs.get_ylim()[0], ymax=axs.get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs.vlines(x=50, ymin=axs.get_ylim()[0], ymax=axs.get_ylim()[1],  colors='white', ls='--', linewidth=1)
plt.xlabel("Time from ripple start (ms)")
plt.ylabel("M-L (µm)")
cbar = plt.colorbar(ax, shrink=.5)
cbar.ax.set_ylabel('Δ spiking per 10 ms', rotation=270, labelpad=8)
cbar.ax.set_label("colorbar_lat") # needed to distinguish colobars

for d in ["left", "top", "bottom", "right"]:
    plt.gca().spines[d].set_visible(False)

axs.set_title("Common ripples - Strong ripples", fontsize=7)

axs.text(0.5, 1.15, "Lateral ripples", transform=axs.transAxes,
      fontsize=10, fontweight='bold', va='top', ha='center')
plt.show()