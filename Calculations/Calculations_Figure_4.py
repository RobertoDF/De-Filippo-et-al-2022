from operator import itemgetter
import dill
import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from Utils.Settings import var_thr, neuropixel_dataset, output_folder_figures_calculations, output_folder_calculations, \
    output_folder_processed_lfps, root_github_repo, waveform_dur_thr
from Utils.Utils import find_ripples_clusters_new, get_ML_limits
from Utils.Utils import spike_summary, clean_ripples_calculations, acronym_to_main_area
from tqdm import tqdm

def consolidate_spiking(spike_clusters, kind):
    sessions_id = []
    summaries = []
    summaries_fraction_active = []
    summaries_fraction_active_by_neuron_type = []

    for session_id in tqdm(set([q[0] for q in spike_clusters.keys()])):
        print(session_id)
        spike_rate = []
        units = spike_clusters[(session_id, 'HPF', 'central')][5]

        for qq in spike_clusters[(session_id, 'HPF', 'central')][1]:
            spike_rate.append([q[1] for q in qq])


        cluster_id = [q[0] for q in spike_clusters[(session_id, 'HPF', 'central')][1][0]]

        spiking_rate_table_medial_seed = pd.DataFrame(spike_rate, columns=pd.MultiIndex.from_arrays(
            [(units.loc[cluster_id, "waveform_duration"] < waveform_dur_thr).map(
            {False: "Putative exc", True: "Putative inh"}), cluster_id], names=["Neuron type", "Cluster id"]))

        spiking_rate_table_medial_seed = spiking_rate_table_medial_seed.loc[:,
                                         (spiking_rate_table_medial_seed != 0).any(axis=0)]

        spike_rate = []
        for qq in spike_clusters[(session_id, 'HPF', 'central')][2]:
            spike_rate.append([q[1] for q in qq])

        cluster_id = [q[0] for q in spike_clusters[(session_id, 'HPF', 'central')][2][0]]

        spiking_rate_table_lateral_seed = pd.DataFrame(spike_rate, columns=pd.MultiIndex.from_arrays(
            [(units.loc[cluster_id, "waveform_duration"] < waveform_dur_thr).map(
            {False: "Putative exc", True: "Putative inh"}), cluster_id], names=["Neuron type", "Cluster id"]))
        spiking_rate_table_lateral_seed = spiking_rate_table_lateral_seed.loc[:,
                                          (spiking_rate_table_lateral_seed != 0).any(axis=0)] # delete cluster with always zero spikes

        sessions_id.append(session_id)

        summary_spiking = pd.concat([spiking_rate_table_lateral_seed.droplevel('Neuron type', axis=1).mean(), spiking_rate_table_medial_seed.droplevel('Neuron type', axis=1).mean(),
                                     spiking_rate_table_lateral_seed.droplevel('Neuron type', axis=1).astype(bool).sum() /\
                                     spiking_rate_table_lateral_seed.droplevel('Neuron type', axis=1).shape[0]
                                        , spiking_rate_table_medial_seed.droplevel('Neuron type', axis=1).astype(bool).sum() /\
                                     spiking_rate_table_medial_seed.droplevel('Neuron type', axis=1).shape[0]], axis=1)
        summary_spiking.columns = ["Lateral seed", "Medial seed", "Active in fraction of ripples with lateral seed (%)",
                                   "Active in fraction of ripples with medial seed (%)"]
        summary_spiking["Session id"] = session_id


        summary_spiking["L-R (µm)"] = units.loc[summary_spiking.index]["left_right_ccf_coordinate"]
        summary_spiking["Post ripple phase"] = kind

        summary_spiking["ecephys_structure_acronym"] = units.loc[summary_spiking.index, "ecephys_structure_acronym"]
        summary_spiking["waveform_duration"] = units.loc[summary_spiking.index,"waveform_duration"]

        summaries.append(summary_spiking)

        _ = pd.DataFrame(
            spiking_rate_table_lateral_seed.astype(bool).sum(axis=1) / spiking_rate_table_lateral_seed.shape[1],
            columns=["Fraction active neurons per ripple (%)"])
        _["Location seed"] = "Lateral seed"
        __ = pd.DataFrame(
            spiking_rate_table_medial_seed.astype(bool).sum(axis=1) / spiking_rate_table_medial_seed.shape[1],
            columns=["Fraction active neurons per ripple (%)"])
        __["Location seed"] = "Medial seed"
        fraction_active_clusters_per_ripples = pd.concat([_, __])
        fraction_active_clusters_per_ripples["Session id"] = session_id
        summaries_fraction_active.append(fraction_active_clusters_per_ripples)

        _ = pd.DataFrame(
            spiking_rate_table_lateral_seed["Putative exc"].astype(bool).sum(axis=1) /
            spiking_rate_table_lateral_seed["Putative exc"].shape[1],
            columns=["Fraction active neurons per ripple (%)"])
        _["Location seed"] = "Lateral seed"
        __ = pd.DataFrame(
            spiking_rate_table_medial_seed["Putative exc"].astype(bool).sum(axis=1) /
            spiking_rate_table_medial_seed["Putative exc"].shape[1],
            columns=["Fraction active neurons per ripple (%)"])
        __["Location seed"] = "Medial seed"
        fraction_active_clusters_per_ripples_by_neuron_type = pd.concat([_, __])
        fraction_active_clusters_per_ripples_by_neuron_type["Session id"] = session_id
        fraction_active_clusters_per_ripples_by_neuron_type["Neuron type"] = "Putative exc"
        summaries_fraction_active_by_neuron_type.append(fraction_active_clusters_per_ripples_by_neuron_type)

        _ = pd.DataFrame(
            spiking_rate_table_lateral_seed["Putative inh"].astype(bool).sum(axis=1) / spiking_rate_table_lateral_seed["Putative inh"].shape[1],
            columns=["Fraction active neurons per ripple (%)"])
        _["Location seed"] = "Lateral seed"
        __ = pd.DataFrame(
            spiking_rate_table_medial_seed["Putative inh"].astype(bool).sum(axis=1) / spiking_rate_table_medial_seed["Putative inh"].shape[1],
            columns=["Fraction active neurons per ripple (%)"])
        __["Location seed"] = "Medial seed"
        fraction_active_clusters_per_ripples_by_neuron_type = pd.concat([_, __])
        fraction_active_clusters_per_ripples_by_neuron_type["Session id"] = session_id
        fraction_active_clusters_per_ripples_by_neuron_type["Neuron type"] = "Putative inh"
        summaries_fraction_active_by_neuron_type.append(fraction_active_clusters_per_ripples_by_neuron_type)

    tot_summary = pd.concat(summaries)  # .mean()
    tot_summary["Location cluster"] = tot_summary.apply(l_r_classifier, axis=1)

    summary_fraction_active_clusters_per_ripples = pd.concat(summaries_fraction_active)
    summary_fraction_active_clusters_per_ripples_by_neuron_type = pd.concat(summaries_fraction_active_by_neuron_type)

    return tot_summary, summary_fraction_active_clusters_per_ripples, summary_fraction_active_clusters_per_ripples_by_neuron_type


def l_m_classifier(row):
    if row["Source M-L (µm)"] < medial_lim_lm:
        v = "Medial"
    elif row["Source M-L (µm)"] > lateral_lim_lm:
        v = "Lateral"
    else:
        v = "Central"
    return v

def l_r_classifier(row):
    if row["L-R (µm)"] < medial_lim:
        v = "Medial"
    elif row["L-R (µm)"] > lateral_lim:
        v = "Lateral"
    else:
        v = "Central"
    return v


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

# calculate session example

session_id_example = 771990200

with open(f'{output_folder_processed_lfps}/lfp_per_probe_{session_id}.pkl', 'rb') as f:
    lfp_per_probe = dill.load(f)

ripples = ripples_calcs[session_id_example][3].copy()

ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)

ripples = ripples.sort_values(by="Start (s)").reset_index(drop=True)

ripples = ripples[ripples["Area"] == "CA1"]

ripples = ripples.reset_index().rename(columns={'index': 'Ripple number'})

try:
    ripples["Z-scored ∫Ripple"] = ripples.groupby("Probe number-area").apply(lambda group:  zscore(group["∫Ripple"], ddof=1)).droplevel(0)
except:
    ripples["Z-scored ∫Ripple"] = ripples.groupby("Probe number-area").apply(lambda group:  zscore(group["∫Ripple"], ddof=1)).T

ripples["Local strong"] = ripples.groupby("Probe number").apply(lambda x: x["∫Ripple"] > x["∫Ripple"].quantile(.9) ).sort_index(level=1).values


source_area = ripples.groupby("Probe number-area").mean()['L-R (µm)'].sub(center).abs().idxmin()

##### I want shaded error bars, thats's why I do this "manually", this function is used under the hood in "get_trajectory_across_time_space"
real_ripple_summary = find_ripples_clusters_new(ripples, source_area)
real_ripple_summary["Location seed"] = real_ripple_summary.apply(l_m_classifier, axis=1)

## example

n = 1721#
idxs_cluster = real_ripple_summary.loc[n][real_ripple_summary.loc[n].index.str.contains("ripple number")].sort_values().dropna()

ripple_cluster_medial_seed = ripples.loc[idxs_cluster]


ml_space = get_ML_limits(var_thr)

colors_idx = round((ripple_cluster_medial_seed["M-L (µm)"] - ml_space.min()) /
                   (ml_space.max() - ml_space.min()) * 255)
colors_idx = colors_idx.astype(int)
ripple_cluster_medial_seed["color index"] = colors_idx
ripple_cluster_medial_seed.sort_values(by="M-L (µm)", inplace=True)

n = 788#841#355
idxs_cluster = real_ripple_summary.loc[n][real_ripple_summary.loc[n].index.str.contains("ripple number")].sort_values().dropna()

ripple_cluster_lateral_seed = ripples.loc[idxs_cluster]
colors_idx = round((ripple_cluster_lateral_seed["M-L (µm)"] - ml_space.min()) /
                   (ml_space.max() - ml_space.min()) * 255)
colors_idx = colors_idx.astype(int)
ripple_cluster_lateral_seed["color index"] = colors_idx
ripple_cluster_lateral_seed.sort_values(by="M-L (µm)", inplace=True)

####

manifest_path = f"{neuropixel_dataset}/manifest.json"

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()

session = cache.get_session_data(session_id_example,
                                 amplitude_cutoff_maximum=np.inf, presence_ratio_minimum=-np.inf, isi_violations_maximum=np.inf)  #, amplitude_cutoff_maximum = np.inf, presence_ratio_minimum = -np.inf, isi_violations_maximum = np.inf)

with open(f'{output_folder_processed_lfps}/lfp_per_probe_{session_id_example}.pkl', 'rb') as f:
    lfp_per_probe = dill.load(f)

lfp = lfp_per_probe[0]

spike_times = session.spike_times

units = session.units

units["parent area"] = units["ecephys_structure_acronym"].apply(lambda area: acronym_to_main_area(area))

target_area = "HPF"
field_to_use_to_compare = "parent area"  # "ecephys_structure_acronym","parent area"
space_sub_spike_times = dict(zip(units[units[field_to_use_to_compare] == target_area].index,
                                 itemgetter(*units[units[field_to_use_to_compare] == target_area].index)(spike_times)))

ripples = ripples_calcs[session_id_example][3].copy() # this is so I brainrender all CA1 locations not only the ones with clear ripples

with open(f"{output_folder_calculations}/spike_clusters_summary_per_parent_area_by_seed_location.pkl", 'rb') as f:
    spike_clusters = dill.load(f)

kind = "Total"
tot_summary,  summary_fraction_active_clusters_per_ripples,  \
summary_fraction_active_clusters_per_ripples_by_neuron_type = consolidate_spiking(spike_clusters, kind)

with open(f"{output_folder_calculations}/spike_clusters_summary_per_parent_area_by_seed_location_after_50ms.pkl", 'rb') as f:
    spike_clusters = dill.load(f)

kind = "Late"
tot_summary_late,  summary_fraction_active_clusters_per_ripples_late, \
summary_fraction_active_clusters_per_ripples_late_by_neuron_type = consolidate_spiking(spike_clusters, kind)

with open(f"{output_folder_calculations}/spike_clusters_summary_per_parent_area_by_seed_location_first_50ms.pkl", 'rb') as f:
    spike_clusters = dill.load(f)

kind = "Early"
tot_summary_early,  summary_fraction_active_clusters_per_ripples_early, \
summary_fraction_active_clusters_per_ripples_early_by_neuron_type = consolidate_spiking(spike_clusters, kind)


with open(f"{ output_folder_figures_calculations}/temp_data_figure_4.pkl", "wb") as fp:
    dill.dump([space_sub_spike_times, target_area, units, field_to_use_to_compare,
               session_id_example, lfp, lfp_per_probe,
               ripple_cluster_lateral_seed, ripple_cluster_medial_seed, source_area, ripples,
               tot_summary_early,  summary_fraction_active_clusters_per_ripples_early,
               summary_fraction_active_clusters_per_ripples_early_by_neuron_type,
               tot_summary_late,  summary_fraction_active_clusters_per_ripples_late,
               summary_fraction_active_clusters_per_ripples_late_by_neuron_type,
               tot_summary, summary_fraction_active_clusters_per_ripples,
               summary_fraction_active_clusters_per_ripples_by_neuron_type], fp)


# create brainrenders
exec(open(f"{root_github_repo}/Figures/Figure_4/Figure_4_brainrender.py").read())



