import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from Utils.Settings import output_folder_calculations
from Utils.Utils import postprocess_spike_hists
from tqdm import tqdm

with open(f"{output_folder_calculations}/spike_hists_summary_per_parent_area_by_seed_location_and_strength.pkl", 'rb') as f:
    spike_hists = dill.load(f)

out = []
for session_id in tqdm([q[0] for q in list(spike_hists.keys()) if q[1] == "HPF"]):
    lrs, _, out_hist_medial = spike_hists[session_id, 'HPF', 'medial']
    lrs, _, out_hist_lateral = spike_hists[session_id, 'HPF', 'lateral']
    print(len(out_hist_medial), len(out_hist_lateral))
    if (len(out_hist_medial) > 5) & (len(out_hist_lateral) > 5):
        means_cut = postprocess_spike_hists(out_hist_lateral, out_hist_medial, lrs)
        means_cut_interp = means_cut.interp(assume_sorted=True, Time_ms=np.arange(-50, 130, .1),
                                            ML=np.arange(1174, 3745, 2),
                                            method="linear")#

        out.append(means_cut_interp)

summary = xr.concat(out, dim="Session")


fig, axs = plt.subplots(1, figsize=(5,5))
ax = summary.mean(dim="Session").diff("Seed").plot(cmap="seismic", add_colorbar=False, vmin=-15, vmax=15)
axs.invert_yaxis()
axs.set_ylim(3600, 1250)
axs.vlines(x=0, ymin=axs.get_ylim()[0], ymax=axs.get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs.vlines(x=50, ymin=axs.get_ylim()[0], ymax=axs.get_ylim()[1],  colors='white', ls='--', linewidth=1)
plt.xlabel("Time from ripple start (ms)")
plt.ylabel("M-L (µm)")
cbar = plt.colorbar(ax, shrink=.5)
cbar.ax.set_ylabel('Δ spiking per 10 ms', rotation=270, labelpad=8)
cbar.ax.set_label("colorbar_exc") # needed to distinguish colobars

for d in ["left", "top", "bottom", "right"]:
    plt.gca().spines[d].set_visible(False)

axs.set_title("Medial seed - Lateral seed", fontsize=7)
plt.show()