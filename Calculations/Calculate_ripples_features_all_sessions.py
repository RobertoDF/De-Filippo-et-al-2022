import pandas as pd
import dill
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from rich import print
import seaborn as sns
from statannotations.Annotator import Annotator
from Utils.Style import palette_ML, palette_timelags
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from time import perf_counter
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.interpolate import interp1d
from Utils.Settings import output_folder_calculations, neuropixel_dataset, var_thr
from Utils.Utils import format_for_annotator, corrfunc,  clean_ripples_calculations, find_ripples_clusters_new
import pingouin as pg

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

    ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)

    ripples = ripples.sort_values(by="Start (s)").reset_index(drop=True)
    ripples = ripples[ripples["Area"] == "CA1"]
    ripples = ripples.reset_index().rename(columns={'index': 'Ripple number'})
   # if any((ripples["L-R (µm)"].unique() < medial_lim)):
    if np.any(ripples["L-R (µm)"].unique() < medial_lim) & np.any(ripples["L-R (µm)"].unique() > lateral_lim) & \
            np.any((ripples["L-R (µm)"].unique() > medial_lim) & (ripples["L-R (µm)"].unique() < lateral_lim)):

        source_area = str(ripples["Probe number"].min()) + "-CA1"
        reference = "Medial"

        try:
            ripples["Z-scored ∫Ripple"] = ripples.groupby("Probe number-area").apply(
                lambda group: zscore(group["∫Ripple"], ddof=1)).droplevel(0)
        except:
            ripples["Z-scored ∫Ripple"] = ripples.groupby("Probe number-area").apply(
                lambda group: zscore(group["∫Ripple"], ddof=1)).T

        ripples["Local strong"] = ripples.groupby("Probe number").apply(
            lambda x: x["∫Ripple"] > x["∫Ripple"].quantile(.9)).sort_index(level=1).values

        inputs.append([ripples, source_area, session_id, reference])

        source_area = str(ripples["Probe number"].max()) + "-CA1"
        reference = "Lateral"

        try:
            ripples["Z-scored ∫Ripple"] = ripples.groupby("Probe number-area").apply(
                lambda group: zscore(group["∫Ripple"], ddof=1)).droplevel(0)
        except:
            ripples["Z-scored ∫Ripple"] = ripples.groupby("Probe number-area").apply(
                lambda group: zscore(group["∫Ripple"], ddof=1)).T

        ripples["Local strong"] = ripples.groupby("Probe number").apply(
            lambda x: x["∫Ripple"] > x["∫Ripple"].quantile(.9)).sort_index(level=1).values

        inputs.append([ripples, source_area, session_id, reference])

        source_area = ripples.groupby("Probe number-area").mean()['L-R (µm)'].sub(center).abs().idxmin()
        reference = "Central"

        try:
            ripples["Z-scored ∫Ripple"] = ripples.groupby("Probe number-area").apply(
                lambda group: zscore(group["∫Ripple"], ddof=1)).droplevel(0)
        except:
            ripples["Z-scored ∫Ripple"] = ripples.groupby("Probe number-area").apply(
                lambda group: zscore(group["∫Ripple"], ddof=1)).T

        ripples["Local strong"] = ripples.groupby("Probe number").apply(
            lambda x: x["∫Ripple"] > x["∫Ripple"].quantile(.9)).sort_index(level=1).values

        inputs.append([ripples, source_area, session_id, reference])


def extract_features(ripples, source_area, session_id, reference):

    real_ripple_summary = find_ripples_clusters_new(ripples, source_area)

    real_ripple_summary["Strong"] = real_ripple_summary["∫Ripple"] > real_ripple_summary["∫Ripple"].quantile(.9)

    _ = real_ripple_summary[real_ripple_summary["Spatial engagement"] > .5]["Source M-L (µm)"].value_counts() /   \
        real_ripple_summary[real_ripple_summary["Spatial engagement"] > .5].shape[0] # detected in at least two probes
    out_by_hip_section = pd.Series()
    out_by_hip_section["Medial"] = _[_.index < medial_lim_lm].sum()
    out_by_hip_section["Lateral"] = _[_.index > lateral_lim_lm].sum()
    out_by_hip_section["Central"] = _[(_.index < lateral_lim_lm) & (_.index > medial_lim_lm)].sum()
    out_seed_ripples = pd.DataFrame(out_by_hip_section, columns=["Percentage seed (%)"]) * 100
    out_seed_ripples["Session id"] = session_id
    out_seed_ripples["Reference"] = reference

    _ = real_ripple_summary[real_ripple_summary["Strong"] == 1]["Source M-L (µm)"].value_counts() / \
             real_ripple_summary[real_ripple_summary["Strong"] == 1].shape[0] # detected in at least two probes
    out_by_hip_strong_section = pd.Series()
    out_by_hip_strong_section["Medial"] = _[_.index < medial_lim_lm].sum()
    out_by_hip_strong_section["Lateral"] = _[_.index > lateral_lim_lm].sum()
    out_by_hip_strong_section["Central"] = _[(_.index < lateral_lim_lm) & (_.index > medial_lim_lm)].sum()
    out_source_strong_ripples = pd.DataFrame(out_by_hip_strong_section, columns=["Percentage seed (%)"]) * 100
    out_source_strong_ripples["Session id"] = session_id
    out_source_strong_ripples["Reference"] = reference

    _ = real_ripple_summary[(real_ripple_summary["Strong"] == 0) & (real_ripple_summary["Spatial engagement"] > .5)]["Source M-L (µm)"].value_counts() / \
             real_ripple_summary[(real_ripple_summary["Strong"] == 0) & (real_ripple_summary["Spatial engagement"] > .5)].shape[0] # detected in at least two probes
    out_by_hip_common_section = pd.Series()
    out_by_hip_common_section["Medial"] = _[_.index < medial_lim_lm].sum()
    out_by_hip_common_section["Lateral"] = _[_.index > lateral_lim_lm].sum()
    out_by_hip_common_section["Central"] = _[(_.index < lateral_lim_lm) & (_.index > medial_lim_lm)].sum()
    out_source_common_ripples = pd.DataFrame(out_by_hip_common_section, columns=["Percentage seed (%)"]) * 100
    out_source_common_ripples["Session id"] = session_id
    out_source_common_ripples["Reference"] = reference

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
    global_strength_strong_ripples = pd.DataFrame([session_id, global_strength_mean, reference], index=["Session id", "Global strength index", "Ripple seed"]).T

    global_strength_mean = real_ripple_summary[real_ripple_summary["∫Ripple"] < real_ripple_summary["∫Ripple"].quantile(.9)][
        "Global strength"].mean()
    global_strength_common_ripples = pd.DataFrame([session_id, global_strength_mean, reference],
                                                  index=["Session id", "Global strength index", "Ripple seed"]).T

    ripples["Local strong"] = ripples.groupby("Probe number").apply(
        lambda x: x["∫Ripple"] > x["∫Ripple"].quantile(.9)).sort_index(level=1).values

    ripples_features = ripples.groupby(["Local strong", "M-L (µm)"])[["Peak frequency (Hz)", "Instantaneous Frequency (Hz)", "Peak power", "Duration (s)", "Amplitude (mV)", "∫Ripple" ]].mean()
    ripples_features["Session id"] = session_id


    percentage_strong_and_source = pd.DataFrame(
        pd.Series((real_ripple_summary[real_ripple_summary["Strong"] == 1]["Source"] == 1).sum() / real_ripple_summary[real_ripple_summary["Strong"]== 1]["Source"].shape[0],
                  name="Percentage source"))
    percentage_strong_and_source["Session id"] = session_id
    percentage_strong_and_source["Reference"] = reference

    def lr_classifier(row):
        if row["L-R (µm)"] < medial_lim:
            v = "Medial"
        elif row["L-R (µm)"] > lateral_lim:
            v = "Lateral"
        else:
            v = "Central"
        return v

    # _ = pd.DataFrame(real_ripple_summary[real_ripple_summary["Spatial engagement"]>0].loc[:, real_ripple_summary.columns.str.contains('Z-scored')].mean(),
    #                  columns=["Z-scored ∫Ripple"]).reset_index()
    # _["L-R (µm)"] = ripples.groupby("Probe number-area").mean()["L-R (µm)"].values
    # _["Location"] = _.apply(lr_classifier, axis=1)
    # strength_across_ML = pd.DataFrame(_.groupby("Location").mean()["Z-scored ∫Ripple"])
    # strength_across_ML["Session id"] = session_id
    # strength_across_ML["Reference"] = reference


    return out_seed_ripples, global_strength_strong_ripples, ripples_features, out_source_common_ripples, \
           percentage_strong_and_source, spatial_engagement, out_source_strong_ripples, \
            global_strength_common_ripples

with Pool(processes=int(len(inputs)/2)) as pool:
    r = pool.starmap_async(extract_features, inputs)
    out = r.get()
    pool.close()

with open(f"{output_folder_calculations}/ripples_features_all_sessions.pkl", "wb") as fp:
    dill.dump(out, fp)


# spatial_engagement_summary = pd.concat([q[5] for q in out])
# spatial_engagement_summary = spatial_engagement_summary.groupby(["Reference", "Spatial engagement"]).mean()["Percentage"].reset_index()
# spatial_engagement_summary.query("Reference == 'Medial'").reset_index(drop=True)["Percentage"].plot(c=palette_ML["Medial"])
# spatial_engagement_summary.query("Reference == 'Central'").reset_index(drop=True)["Percentage"].plot(c=palette_ML["Central"])
# spatial_engagement_summary.query("Reference == 'Lateral'").reset_index(drop=True)["Percentage"].plot(c=palette_ML["Lateral"])
# plt.show()




