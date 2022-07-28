import dill
from Utils.Utils import clean_ripples_calculations
from Utils.Settings import clip_ripples_clusters, var_thr, output_folder_figures_calculations, output_folder_calculations, minimum_ripples_count_generated_in_lateral_or_medial_spike_analysis
from scipy.stats import sem
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
import pandas as pd
from scipy.stats import zscore
from Figures.Figure_4.Figure_4_clusters_per_ripple_early_late import ttest_early_clus_per_ripple, ttest_late_clus_per_ripple
from Figures.Figure_4.Figure_4_spiking_rate_early_late import ttest_late_spiking, ttest_early_spiking

with open(f'{output_folder_calculations}/ripples_features_sessions_across_all_ML.pkl', 'rb') as f:
    out = dill.load(f)
seed_ripples_by_hip_section_summary_strong = pd.concat([q[6] for q in out])
seed_ripples_by_hip_section_summary_strong = seed_ripples_by_hip_section_summary_strong.reset_index().rename(columns={'index': 'Location seed'})

out_test_seed_ripples_by_hip_section_summary_strong = pd.DataFrame(seed_ripples_by_hip_section_summary_strong[seed_ripples_by_hip_section_summary_strong["Reference"]=="Central"]\
            .pairwise_tukey(effsize="cohen", dv="Percentage seed (%)", between="Location seed"))[["A","B", "p-tukey"]]
p_val_medial_lateral = round(out_test_seed_ripples_by_hip_section_summary_strong.query("B == 'Medial seed' & A == 'Lateral seed'")['p-tukey'].values[0], 5)

with open(f"{output_folder_figures_calculations}/temp_data_supp_figure_1.pkl", 'rb') as f:
    data_area, ripple_power, session_id, session_summary, summary_table, ripples = dill.load(f)

with open(f'{output_folder_calculations}/trajectories_by_strength.pkl', 'rb') as f:
    trajectories_by_strength = dill.load(f)

with open(f'{output_folder_calculations}/sessions_features.pkl', 'rb') as f:
    sessions_durations, sessions_durations_quiet, sessions_durations_running = dill.load(f)

with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)


with open(f"{output_folder_figures_calculations}/temp_data_figure_1.pkl", "rb") as fp:  # Unpickling
    sessions, high_distance, low_distance, ripples_lags, ripples_lags_inverted_reference, ripples_calcs, summary_corrs = dill.load(fp)

_ = summary_corrs[(summary_corrs["Comparison"]=="CA1-CA1")]

y = _["Correlation"]
x = _["Distance (µm)"]

r, _ = pearsonr(x, y)
r_squared_corr_distance = round(r**2,4)

ripple_freq_total = {}
for session_id,_ in ripples_calcs.items():
    selected_probe = _[5]
    ripples = _[3]
    ripple_freq_total[session_id] = ripples[ripples["Probe number"]==selected_probe].shape[0]/(sessions_durations[session_id])


quartiles_distance = summary_corrs[summary_corrs["Comparison"] == "CA1-CA1"]["Distance (µm)"].quantile(
        [0.25, 0.5, 0.75])
quartiles_correlation = summary_corrs[summary_corrs["Comparison"] == "CA1-CA1"]["Correlation"].quantile([0.25, 0.5, 0.75])

ripples_lags_clipped = ripples_lags[ripples_lags["Lag (ms)"].between(clip_ripples_clusters[0], clip_ripples_clusters[1])]


quant = 0.9
ripples_lags_strong = ripples_lags_clipped.groupby(['Session', "Type"]).apply(lambda group: group[group["∫Ripple"] >= group["∫Ripple"].quantile(quant)]).reset_index(drop=True)
ripples_lags_strong["Quantile"] = "Strong ripples"

ripples_lags_weak = ripples_lags_clipped.groupby(['Session', "Type"]).apply(lambda group: group[group["∫Ripple"] < group["∫Ripple"].quantile(quant)]).reset_index(drop=True)
ripples_lags_weak["Quantile"] = "Common ripples"

ripples_lags_by_percentile = pd.concat([ripples_lags_strong, ripples_lags_weak])
ripples_lags_by_percentile["Reference"] = "Medial"

ripples_lags_inverted_reference_clipped = ripples_lags_inverted_reference[ripples_lags_inverted_reference["Lag (ms)"].between(clip_ripples_clusters[0], clip_ripples_clusters[1])]

quant = 0.9
ripples_lags_inverted_reference_strong = ripples_lags_inverted_reference_clipped.groupby(['Session', "Type"]).apply(lambda group: group[group["∫Ripple"] > group["∫Ripple"].quantile(quant)]).reset_index(drop=True)
ripples_lags_inverted_reference_strong["Quantile"] = "Strong ripples"

ripples_lags_inverted_reference_weak = ripples_lags_inverted_reference_clipped.groupby(['Session', "Type"]).apply(lambda group: group[group["∫Ripple"] < group["∫Ripple"].quantile(quant)]).reset_index(drop=True)
ripples_lags_inverted_reference_weak["Quantile"] = "Common ripples"

ripples_lags_inverted_reference_by_percentile = pd.concat([ripples_lags_inverted_reference_strong, ripples_lags_inverted_reference_weak])
ripples_lags_inverted_reference_by_percentile["Reference"] = "Lateral"

summary_lags = pd.concat([ripples_lags_by_percentile, ripples_lags_inverted_reference_by_percentile])

data = summary_lags[summary_lags["Type"] == "High distance (µm)"]

data_reference_medial = data[data["Reference"]=="Medial"]

fig1_mean_lag_strong_ref_medial = np.round(data_reference_medial[data_reference_medial["Quantile"]== "Strong ripples"].groupby("Session").mean()["Lag (ms)"].mean(), 2)
fig1_mean_lag_common_ref_medial = np.round(data_reference_medial[data_reference_medial["Quantile"]== "Common ripples"].groupby("Session").mean()["Lag (ms)"].mean(), 2)
fig1_sem_lag_strong_ref_medial = np.round(data_reference_medial[data_reference_medial["Quantile"]== "Strong ripples"].groupby("Session").mean()["Lag (ms)"].sem(), 2)
fig1_sem_lag_common_ref_medial = np.round(data_reference_medial[data_reference_medial["Quantile"]== "Common ripples"].groupby("Session").mean()["Lag (ms)"].sem(), 2)

data_reference_lateral = data[data["Reference"]=="Lateral"]

fig1_mean_lag_strong_ref_lateral = np.round(data_reference_lateral[data_reference_lateral["Quantile"]== "Strong ripples"].groupby("Session").mean()["Lag (ms)"].mean(), 2)
fig1_mean_lag_common_ref_lateral = np.round(data_reference_lateral[data_reference_lateral["Quantile"]== "Common ripples"].groupby("Session").mean()["Lag (ms)"].mean(), 2)
fig1_sem_lag_strong_ref_lateral = np.round(data_reference_lateral[data_reference_lateral["Quantile"]== "Strong ripples"].groupby("Session").mean()["Lag (ms)"].sem(), 2)
fig1_sem_lag_common_ref_lateral = np.round(data_reference_lateral[data_reference_lateral["Quantile"]== "Common ripples"].groupby("Session").mean()["Lag (ms)"].sem(), 2)

data_common = data[data["Quantile"]=="Common ripples"]

fig1_mean_lag_ref_medial_common = np.round(data_common[data_common["Reference"]== "Medial"].groupby("Session").mean()["Lag (ms)"].mean(), 2)
fig1_mean_lag_ref_lateral_common = np.round(data_common[data_common["Reference"]== "Lateral"].groupby("Session").mean()["Lag (ms)"].mean(), 2)
fig1_sem_lag_ref_medial_common = np.round(data_common[data_common["Reference"]== "Medial"].groupby("Session").mean()["Lag (ms)"].sem(), 2)
fig1_sem_lag_ref_lateral_common  = np.round(data_common[data_common["Reference"]== "Lateral"].groupby("Session").mean()["Lag (ms)"].sem(), 2)

data_strong = data[data["Quantile"]=="Strong ripples"]

fig1_mean_lag_ref_medial_strong = np.round(data_strong[data_strong["Reference"]== "Medial"].groupby("Session").mean()["Lag (ms)"].mean(), 2)
fig1_mean_lag_ref_lateral_strong = np.round(data_strong[data_strong["Reference"]== "Lateral"].groupby("Session").mean()["Lag (ms)"].mean(), 2)
fig1_sem_lag_ref_medial_strong = np.round(data_strong[data_strong["Reference"]== "Medial"].groupby("Session").mean()["Lag (ms)"].sem(), 2)
fig1_sem_lag_ref_lateral_strong = np.round(data_strong[data_strong["Reference"]== "Lateral"].groupby("Session").mean()["Lag (ms)"].sem(), 2)

with open(f"{output_folder_calculations}/number_ripples_per_session_best_CA1_channel.pkl", 'rb') as f:
   number_ripples_per_session_best_CA1_channel = dill.load(f)

with open(f'{output_folder_calculations}/ripples_features_sessions_across_all_ML.pkl', 'rb') as f:
    out = dill.load(f)


ripples_features_summary = pd.concat([q[2] for q in out]).drop_duplicates()
ripples_features_summary = ripples_features_summary.reset_index().rename(columns={'index': 'M-L (µm)'})
ripples_features_summary.rename(columns={"Local strong":"Type"}, inplace=True)

ripples_features_summary["Type"] = ripples_features_summary["Type"].astype("category").cat.rename_categories({False:"Common ripples", True:"Strong ripples"})

y = ripples_features_summary[ripples_features_summary["Type"]=="Strong ripples"]["Duration (s)"]
x = ripples_features_summary[ripples_features_summary["Type"]=="Strong ripples"]["M-L (µm)"]
r, _ = pearsonr(x, y)
r_strong = round(r ** 2, 3)

y = ripples_features_summary[ripples_features_summary["Type"]=="Common ripples"]["Duration (s)"]
x = ripples_features_summary[ripples_features_summary["Type"]=="Common ripples"]["M-L (µm)"]
r, _ = pearsonr(x, y)
r_common = round(r ** 2, 3)

with open(f"{output_folder_calculations}/spike_hists_summary_per_parent_area_by_seed_location.pkl", 'rb') as f:
    spike_hists = dill.load(f)

out = []
for session_id, sel in tqdm(ripples_calcs.items()):
    ripples = ripples_calcs[session_id][3].copy()
    ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)

    ripples["Z-scored ∫Ripple"] = ripples["∫Ripple"].transform(lambda x: zscore(x, ddof=1))
    ripples["Z-scored amplitude (mV)"] = ripples["Amplitude (mV)"].transform(lambda x: zscore(x, ddof=1))
    ripples["Session"] = session_id

    out.append(ripples.groupby("Probe number").mean())

data = pd.concat(out)
data["Session"] = data["Session"].astype("category")
data.reset_index(inplace=True)

r_ML_amp, _ = pearsonr(data["Z-scored amplitude (mV)"], data["M-L (µm)"])
r_ML_strength, _ = pearsonr(data["Z-scored ∫Ripple"], data["M-L (µm)"])

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

