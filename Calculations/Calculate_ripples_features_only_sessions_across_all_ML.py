from multiprocessing import Pool
import dill
import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from rich import print
from scipy.interpolate import interp1d
from scipy.stats import zscore
from tqdm import tqdm
from Utils.Settings import output_folder_calculations, neuropixel_dataset, var_thr
from Utils.Utils import clean_ripples_calculations, find_ripples_clusters_new

manifest_path = f"{neuropixel_dataset}/manifest.json"

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()

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
center_lm = center - 5691.510009765625


inputs = []
for session_id in tqdm(ripples_calcs.keys()):
    print(session_id)
    ripples = ripples_calcs[session_id][3].copy()

    sel_probe = ripples_calcs[session_id][5]

    if ripples[ripples['Probe number'] == sel_probe].shape[0] < 1000:
        continue

    ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)

    ripples = ripples.sort_values(by="Start (s)").reset_index(drop=True)
    ripples = ripples[ripples["Area"] == "CA1"]
    ripples = ripples.reset_index().rename(columns={'index': 'Ripple number'})
   # if any((ripples["L-R (µm)"].unique() < medial_lim)):
    if np.any(ripples["L-R (µm)"].unique() < medial_lim) & np.any(ripples["L-R (µm)"].unique() > lateral_lim) & \
            np.any((ripples["L-R (µm)"].unique() > medial_lim) & (ripples["L-R (µm)"].unique() < lateral_lim)):

        seed_area = str(ripples["Probe number"].min()) + "-CA1"
        reference = "Medial"

        try:
            ripples["Z-scored ∫Ripple"] = ripples.groupby("Probe number-area").apply(
                lambda group: zscore(group["∫Ripple"], ddof=1)).droplevel(0)
        except:
            ripples["Z-scored ∫Ripple"] = ripples.groupby("Probe number-area").apply(
                lambda group: zscore(group["∫Ripple"], ddof=1)).T

        ripples["Local strong"] = ripples.groupby("Probe number").apply(
            lambda x: x["∫Ripple"] > x["∫Ripple"].quantile(.9)).sort_index(level=1).values

        inputs.append([ripples, seed_area, session_id, reference])

        seed_area = str(ripples["Probe number"].max()) + "-CA1"
        reference = "Lateral"

        inputs.append([ripples, seed_area, session_id, reference])

        seed_area = ripples.groupby("Probe number-area").mean()['L-R (µm)'].sub(center).abs().idxmin()
        reference = "Central"

        inputs.append([ripples, seed_area, session_id, reference])

def lr_classifier(row):
    if row["L-R (µm)"] < medial_lim:
        v = "Medial"
    elif row["L-R (µm)"] > lateral_lim:
        v = "Lateral"
    else:
        v = "Central"
    return v

def extract_features(ripples, seed_area, session_id, reference):

    real_ripple_summary = find_ripples_clusters_new(ripples, seed_area)

    real_ripple_summary["Strong"] = real_ripple_summary["∫Ripple"] > real_ripple_summary["∫Ripple"].quantile(.9)
    real_ripple_summary["Long"] = real_ripple_summary["∫Ripple"] > real_ripple_summary["Duration (s)"].quantile(.9)

    _ = real_ripple_summary[real_ripple_summary["Spatial engagement"] > .5]["Source M-L (µm)"].value_counts() /   \
        real_ripple_summary[real_ripple_summary["Spatial engagement"] > .5].shape[0] # detected in at least half of the probes
    out_by_hip_section = pd.Series()
    out_by_hip_section["Medial seed"] = _[_.index < medial_lim_lm].sum()
    out_by_hip_section["Lateral seed"] = _[_.index > lateral_lim_lm].sum()
    out_by_hip_section["Central seed"] = _[(_.index < lateral_lim_lm) & (_.index > medial_lim_lm)].sum()
    out_seed_ripples = pd.DataFrame(out_by_hip_section, columns=["Percentage seed (%)"]) * 100
    out_seed_ripples["Session id"] = session_id
    out_seed_ripples["Reference"] = reference

    _ = real_ripple_summary[real_ripple_summary["Strong"] == 1]["Source M-L (µm)"].value_counts() / \
             real_ripple_summary[real_ripple_summary["Strong"] == 1].shape[0]
    out_by_hip_strong_section = pd.Series()
    out_by_hip_strong_section["Medial seed"] = _[_.index < medial_lim_lm].sum()
    out_by_hip_strong_section["Lateral seed"] = _[_.index > lateral_lim_lm].sum()
    out_by_hip_strong_section["Central seed"] = _[(_.index < lateral_lim_lm) & (_.index > medial_lim_lm)].sum()
    out_seed_strong_ripples = pd.DataFrame(out_by_hip_strong_section, columns=["Percentage seed (%)"]) * 100
    out_seed_strong_ripples["Session id"] = session_id
    out_seed_strong_ripples["Reference"] = reference

    _ = real_ripple_summary[real_ripple_summary["Long"] == 1]["Source M-L (µm)"].value_counts() / \
             real_ripple_summary[real_ripple_summary["Long"] == 1].shape[0]
    out_by_hip_duration_section = pd.Series()
    out_by_hip_duration_section["Medial seed"] = _[_.index < medial_lim_lm].sum()
    out_by_hip_duration_section["Lateral seed"] = _[_.index > lateral_lim_lm].sum()
    out_by_hip_duration_section["Central seed"] = _[(_.index < lateral_lim_lm) & (_.index > medial_lim_lm)].sum()
    out_seed_long_ripples = pd.DataFrame(out_by_hip_duration_section, columns=["Percentage seed (%)"]) * 100
    out_seed_long_ripples["Session id"] = session_id
    out_seed_long_ripples["Reference"] = reference
    
    _ = real_ripple_summary[(real_ripple_summary["Strong"] == 0) & (real_ripple_summary["Spatial engagement"] > .5)]["Source M-L (µm)"].value_counts() / \
             real_ripple_summary[(real_ripple_summary["Strong"] == 0) & (real_ripple_summary["Spatial engagement"] > .5)].shape[0] # detected in at least two probes
    out_by_hip_common_section = pd.Series()
    out_by_hip_common_section["Medial seed"] = _[_.index < medial_lim_lm].sum()
    out_by_hip_common_section["Lateral seed"] = _[_.index > lateral_lim_lm].sum()
    out_by_hip_common_section["Central seed"] = _[(_.index < lateral_lim_lm) & (_.index > medial_lim_lm)].sum()
    out_seed_common_ripples = pd.DataFrame(out_by_hip_common_section, columns=["Percentage seed (%)"]) * 100
    out_seed_common_ripples["Session id"] = session_id
    out_seed_common_ripples["Reference"] = reference

    _ = (real_ripple_summary["Spatial engagement"].value_counts() / real_ripple_summary.shape[0]).sort_index()
    y = _.values
    x = _.index

    f = interp1d(x, y, kind="cubic")
    xnew = np.arange(0, 1, 0.01)
    ynew = f(xnew)

    spatial_engagement = pd.DataFrame([xnew, ynew], index=["Spatial engagement", "Percentage"]).T

    spatial_engagement["Session id"] = session_id
    spatial_engagement["Reference"] = reference

    global_strength_mean = real_ripple_summary[real_ripple_summary["∫Ripple"] > real_ripple_summary["∫Ripple"].quantile(.9)]["Global strength"].mean()

    global_strength_strong_ripples = pd.DataFrame([session_id, global_strength_mean, reference],
                                                  index=["Session id", "Strength conservation index", "Reference"]).T

    global_strength_mean = real_ripple_summary[real_ripple_summary["∫Ripple"] < real_ripple_summary["∫Ripple"].quantile(.9)][
        "Global strength"].mean()

    global_strength_common_ripples = pd.DataFrame([session_id, global_strength_mean, reference],
                                                  index=["Session id", "Strength conservation index", "Reference"]).T

    ripples["Local strong"] = ripples.groupby("Probe number").apply(
        lambda x: x["∫Ripple"] > x["∫Ripple"].quantile(.9)).sort_index(level=1).values

    ripples_features = ripples.groupby(["Local strong", "M-L (µm)"])[["Peak frequency (Hz)", "Instantaneous Frequency (Hz)", "Peak power", "Duration (s)", "Amplitude (mV)", "∫Ripple" ]].mean()
    ripples_features["Session id"] = session_id

    percentage_strong_and_seed = pd.DataFrame(
        pd.Series((real_ripple_summary[real_ripple_summary["Strong"] == 1]["Source"] == 1).sum() / real_ripple_summary[real_ripple_summary["Strong"]== 1]["Source"].shape[0],
                  name="Percentage seed"))
    percentage_strong_and_seed["Session id"] = session_id


    ripples["Location seed"] = ripples.apply(lr_classifier, axis=1)
    count_detected_ripples = pd.DataFrame(real_ripple_summary[real_ripple_summary["Spatial engagement"] > .5].loc[:,
                                          real_ripple_summary.columns.str.contains('lag')].count(),
                                          columns=["Count detected ripples"])
    idx = [q[0] for q in count_detected_ripples.index.str.split()]
    locations = [ripples[ripples["Probe number-area"] == q]["Location seed"].iloc[0] for q in idx]
    count_detected_ripples["Location seed"] = locations
    count_detected_ripples = count_detected_ripples.groupby("Location seed").mean()
    count_detected_ripples["Session id"] = session_id
    count_detected_ripples["Reference"] = reference

    return out_seed_ripples, global_strength_strong_ripples, ripples_features, out_seed_common_ripples, \
           percentage_strong_and_seed, spatial_engagement, out_seed_strong_ripples, \
            global_strength_common_ripples, count_detected_ripples, out_seed_long_ripples


with Pool(processes=int(len(inputs)/2)) as pool:
    r = pool.starmap_async(extract_features, inputs)
    out = r.get()
    pool.close()

with open(f"{output_folder_calculations}/ripples_features_sessions_across_all_ML.pkl", "wb") as fp:
    dill.dump(out, fp)







