import matplotlib.pyplot as plt
import pylustrator
from Utils.Settings import output_folder_supplementary, Adapt_for_Nature_style
from Utils.Utils import Naturize
import Utils.Style
import dill
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from Utils.Settings import output_folder_calculations
from Utils.Utils import postprocess_spike_hists, postprocess_spike_hists_strength
from tqdm import tqdm

pylustrator.start()

with open(f"{output_folder_calculations}/spike_hists_summary_per_parent_area_by_seed_location_exc.pkl", 'rb') as f:
    spike_hists = dill.load(f)

out = []
for session_id in tqdm([q[0] for q in list(spike_hists.keys()) if q[1] == "HPF"]):
    lrs, out_hist_medial, out_hist_lateral = spike_hists[session_id, 'HPF', 'central']
    print(session_id, len(out_hist_medial) > 0, len(out_hist_lateral) > 0)
    if (len(out_hist_medial) > 0) & (len(out_hist_lateral) > 0):
        means_cut = postprocess_spike_hists(out_hist_lateral, out_hist_medial, lrs)
        means_cut_interp = means_cut.interp(assume_sorted=True, Time_ms=np.arange(-50, 130, .1),
                                            ML=np.arange(1174, 3745, 2),
                                            method="linear")#

        out.append(means_cut_interp)

summary = xr.concat(out, dim="Session")

fig, axs = plt.subplots(3,2, figsize=(5,5))
ax = summary.mean(dim="Session").diff("Seed").plot(ax=axs[0,0], cmap="seismic", add_colorbar=True, vmin=-10, vmax=10)
axs[0,0].invert_yaxis()
axs[0,0].set_ylim(3600, 1250)
axs[0,0].vlines(x=0, ymin=axs[0,0].get_ylim()[0], ymax=axs[0,0].get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs[0,0].vlines(x=50, ymin=axs[0,0].get_ylim()[0], ymax=axs[0,0].get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs[0,0].set_xlabel("Time from ripple start (ms)")
axs[0,0].set_ylabel("M-L (µm)")



axs[0,0].set_title("Medial seed - Lateral seed", fontsize=7)



with open(f"{output_folder_calculations}/spike_hists_summary_per_parent_area_by_seed_location_inh.pkl", 'rb') as f:
    spike_hists = dill.load(f)

out = []
for session_id in tqdm([q[0] for q in list(spike_hists.keys()) if q[1] == "HPF"]):
    lrs, out_hist_medial, out_hist_lateral = spike_hists[session_id, 'HPF', 'central']
    print(session_id, len(out_hist_medial) > 0, len(out_hist_lateral) > 0)
    if (len(out_hist_medial) > 0) & (len(out_hist_lateral) > 0):
        means_cut = postprocess_spike_hists(out_hist_lateral, out_hist_medial, lrs)
        means_cut_interp = means_cut.interp(assume_sorted=True, Time_ms=np.arange(-50, 130, .1),
                                            ML=np.arange(1174, 3745, 2),
                                            method="linear")

        out.append(means_cut_interp)

summary = xr.concat(out, dim="Session")


ax = summary.mean(dim="Session").diff("Seed").plot(ax=axs[0,1], cmap="seismic", add_colorbar=True, vmin=-10, vmax=10)
axs[0,1].invert_yaxis()
axs[0,1].set_ylim(3600, 1250)
axs[0,1].vlines(x=0, ymin=axs[0,1].get_ylim()[0], ymax=axs[0,1].get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs[0,1].vlines(x=50, ymin=axs[0,1].get_ylim()[0], ymax=axs[0,1].get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs[0,1].set_xlabel("Time from ripple start (ms)")
axs[0,1].set_ylabel("M-L (µm)")



axs[0,1].set_title("Medial seed - Lateral seed", fontsize=7)




with open(f"{output_folder_calculations}/spike_hists_summary_per_parent_area_by_seed_location_and_strength.pkl", 'rb') as f:
    spike_hists = dill.load(f)

out = []
for session_id in tqdm([q[0] for q in list(spike_hists.keys()) if q[1] == "HPF"]):
    lrs, out_hist_strong, out_hist_common = spike_hists[session_id, 'HPF', 'medial']
    print(session_id, len(out_hist_strong) > 0, len(out_hist_common) > 0)
    if (len(out_hist_strong) > 5) & (len(out_hist_common) > 5):
        means_cut = postprocess_spike_hists_strength(out_hist_common, out_hist_strong, lrs)
        means_cut_interp = means_cut.interp(assume_sorted=True, Time_ms=np.arange(-50, 130, .1),
                                            ML=np.arange(1174, 3745, 2),
                                            method="linear")#

        out.append(means_cut_interp)

summary = xr.concat(out, dim="Session")

ax = summary.mean(dim="Session").diff("Strength").plot(ax=axs[1,0], cmap="seismic", add_colorbar=True, vmin=-20, vmax=20)
axs[1,0].invert_yaxis()
axs[1,0].set_ylim(3600, 1250)
axs[1,0].vlines(x=0, ymin=axs[1,0].get_ylim()[0], ymax=axs[1,0].get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs[1,0].vlines(x=50, ymin=axs[1,0].get_ylim()[0], ymax=axs[1,0].get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs[1,0].set_xlabel("Time from ripple start (ms)")
axs[1,0].set_ylabel("M-L (µm)")



axs[1,0].set_title("Common ripples - Strong ripples", fontsize=7)


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

ax = summary.mean(dim="Session").diff("Strength").plot(ax=axs[1,1], cmap="seismic", add_colorbar=True, vmin=-20, vmax=20)
axs[1,1].invert_yaxis()
axs[1,1].set_ylim(3600, 1250)
axs[1,1].vlines(x=0, ymin=axs[1,1].get_ylim()[0], ymax=axs[1,1].get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs[1,1].vlines(x=50, ymin=axs[1,1].get_ylim()[0], ymax=axs[1,1].get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs[1,1].set_xlabel("Time from ripple start (ms)")
axs[1,1].set_ylabel("M-L (µm)")



axs[1,1].set_title("Common ripples - Strong ripples", fontsize=7)




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


ax = summary.mean(dim="Session").diff("Seed").plot(ax=axs[2,0], cmap="seismic", add_colorbar=True, vmin=-15, vmax=15)
axs[2,0].invert_yaxis()
axs[2,0].set_ylim(3600, 1250)
axs[2,0].vlines(x=0, ymin=axs[2,0].get_ylim()[0], ymax=axs[2,0].get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs[2,0].vlines(x=50, ymin=axs[2,0].get_ylim()[0], ymax=axs[2,0].get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs[2,0].set_xlabel("Time from ripple start (ms)")
axs[2,0].set_ylabel("M-L (µm)")


axs[2,0].set_title("Medial seed - Lateral seed", fontsize=7)



out = []
for session_id in tqdm([q[0] for q in list(spike_hists.keys()) if q[1] == "HPF"]):
    lrs, out_hist_medial, _ = spike_hists[session_id, 'HPF', 'medial']
    lrs, out_hist_lateral, _ = spike_hists[session_id, 'HPF', 'lateral']
    if (len(out_hist_medial) > 5) & (len(out_hist_lateral) > 5):
        means_cut = postprocess_spike_hists(out_hist_lateral, out_hist_medial, lrs)
        means_cut_interp = means_cut.interp(assume_sorted=True, Time_ms=np.arange(-50, 130, .1),
                                            ML=np.arange(1174, 3745, 2),
                                            method="linear")#

        out.append(means_cut_interp)

summary = xr.concat(out, dim="Session")

ax = summary.mean(dim="Session").diff("Seed").plot(ax=axs[2,1], cmap="seismic", add_colorbar=True, vmin=-15, vmax=15)
axs[2,1].invert_yaxis()
axs[2,1].set_ylim(3600, 1250)
axs[2,1].vlines(x=0, ymin=axs[2,1].get_ylim()[0], ymax=axs[2,1].get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs[2,1].vlines(x=50, ymin=axs[2,1].get_ylim()[0], ymax=axs[2,1].get_ylim()[1],  colors='white', ls='--', linewidth=1)
axs[2,1].set_xlabel("Time from ripple start (ms)")
axs[2,1].set_ylabel("M-L (µm)")



axs[2,1].set_title("Medial seed - Lateral seed", fontsize=7)


if Adapt_for_Nature_style is True:
    Naturize()

fig.tight_layout()
plt.subplots_adjust(hspace = 0.5)
#% end: automatic generated code from pylustrator
plt.show()

#plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_10", dpi=300)
