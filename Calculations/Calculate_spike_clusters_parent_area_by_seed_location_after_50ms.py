import pandas as pd
import dill
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from rich import print
from tqdm.auto import tqdm
from time import perf_counter
from scipy.stats import zscore
from Utils.Settings import window_spike_clu_late, output_folder_calculations, neuropixel_dataset, var_thr, minimum_ripples_count_spike_analysis, minimum_ripples_count_generated_in_lateral_or_medial_spike_analysis
from Utils.Utils import acronym_to_main_area, clean_ripples_calculations, find_ripples_clusters_new, process_spikes_per_ripple,\
        batch_process_spike_clusters_by_seed_location

t1_start = perf_counter()

manifest_path = f"{neuropixel_dataset}/manifest.json"

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()
#ProfileReport(sessions)

with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)

spike_clusters = {}

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

def l_m_classifier(row):
    if row["Source M-L (µm)"] < medial_lim_lm:
        v = "Medial"
    elif row["Source M-L (µm)"] > lateral_lim_lm:
        v = "Lateral"
    else:
        v = "Central"
    return v

func = process_spikes_per_ripple

for session_id in tqdm(ripples_calcs.keys()):
    print(session_id)

    session = cache.get_session_data(session_id,
                                     amplitude_cutoff_maximum=np.inf, presence_ratio_minimum=-np.inf,
                                     isi_violations_maximum=np.inf)  # , amplitude_cutoff_maximum = np.inf, presence_ratio_minimum = -np.inf, isi_violations_maximum = np.inf)

    units = session.units
    units["parent area"] = units["ecephys_structure_acronym"].apply(lambda area: acronym_to_main_area(area))


    # #  each area, change output name accordingly
    areas = np.delete(units["parent area"].unique(),
                      np.argwhere(units["parent area"].unique() == "grey"))  # delete grey if present

    if "HPF" in areas:
        areas = ["HPF"]
    else:
        continue

    print(f"In session {session_id} areas recorded: {areas}")

    spike_times = session.spike_times

    ripples = ripples_calcs[session_id][3].copy()

    sel_probe = ripples_calcs[session_id][5]

    print(f"number ripples on best probe: {ripples[ripples['Probe number'] == sel_probe].shape[0]}")

    if ripples[ripples['Probe number'] == sel_probe].shape[0] < minimum_ripples_count_spike_analysis:
        continue


    ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)

    ripples = ripples.sort_values(by="Start (s)").reset_index(drop=True)
    ripples = ripples[ripples["Area"] == "CA1"]
    ripples = ripples.reset_index().rename(columns={'index': 'Ripple number'})

    print(session_id, "Recording in each ML section:",
          np.any(ripples["L-R (µm)"].unique() < medial_lim) & np.any(ripples["L-R (µm)"].unique() > lateral_lim) & \
          np.any((ripples["L-R (µm)"].unique() > medial_lim) & (ripples["L-R (µm)"].unique() < lateral_lim)))

    if np.any(ripples["L-R (µm)"].unique() < medial_lim) & np.any(ripples["L-R (µm)"].unique() > lateral_lim) & \
            np.any((ripples["L-R (µm)"].unique() > medial_lim) & (ripples["L-R (µm)"].unique() < lateral_lim)) == False:
        continue

    ripples["Local strong"] = ripples.groupby("Probe number").apply(
        lambda x: x["∫Ripple"] > x["∫Ripple"].quantile(.9)).sort_index(level=1).values

    try:
        ripples["Z-scored ∫Ripple"] = ripples.groupby("Probe number-area").apply(
            lambda group: zscore(group["∫Ripple"], ddof=1)).droplevel(0)
    except:
        ripples["Z-scored ∫Ripple"] = ripples.groupby("Probe number-area").apply(
            lambda group: zscore(group["∫Ripple"], ddof=1)).T

    to_loop = []
    to_loop.append((ripples.groupby("Probe number-area").mean()['L-R (µm)'].sub(center).abs().idxmin(), "central"))

    print(f"In session {session_id} process:{to_loop}")

    if ripples.shape[0] > 0:
        for source_area, type_source in to_loop:

            real_ripple_summary = find_ripples_clusters_new(ripples, source_area)
            real_ripple_summary = real_ripple_summary[real_ripple_summary["Spatial engagement"] > .5]
            real_ripple_summary["Location seed"] = real_ripple_summary.apply(l_m_classifier, axis=1)

            print(
                f"in {session_id}, medial ripples number: {real_ripple_summary[real_ripple_summary['Location seed'] == 'Medial'].shape[0]}, " \
                f"lateral ripples number: {real_ripple_summary[real_ripple_summary['Location seed'] == 'Lateral'].shape[0]}")
            if (real_ripple_summary[real_ripple_summary["Location seed"] == "Medial"].shape[0] < minimum_ripples_count_generated_in_lateral_or_medial_spike_analysis) or \
                    (real_ripple_summary[real_ripple_summary["Location seed"] == "Lateral"].shape[0] < minimum_ripples_count_generated_in_lateral_or_medial_spike_analysis):
                continue

            field_to_use_to_compare = "parent area"  # "ecephys_structure_acronym","parent area"
            n_cpus = 60

            for target_area in tqdm(areas):

                lrs, out_clusters_medial, out_clusters_lateral, duration_ripples_medial, duration_ripples_lateral, units\
                    = batch_process_spike_clusters_by_seed_location(func, real_ripple_summary, units, spike_times,
                                                                           target_area, field_to_use_to_compare,
                                                                    n_cpus, window_spike_clu_late)

                spike_clusters[(session_id, target_area, type_source)] = [lrs, out_clusters_medial, out_clusters_lateral,
                                                                          duration_ripples_medial, duration_ripples_lateral, units]

with open(f"{output_folder_calculations}/spike_clusters_summary_per_parent_area_by_seed_location_after_50ms.pkl", "wb") as fp:
    dill.dump(spike_clusters, fp)

t1_stop = perf_counter()

print("Elapsed time during the whole program in seconds:",
      t1_stop - t1_start)