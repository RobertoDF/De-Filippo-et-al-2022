import dill
import numpy as np
from Utils.Settings import output_folder_calculations, output_folder_figures_calculations
import pandas as pd
from scipy.stats import pearsonr

# variables figure 1
from Figures.Figure_1.Figure_1_pointplot_lag import summary_lags
from Figures.Figure_1.Figure_1_pointplot_lag_summary import p_value_ttest_lag, p_value_ttest_abs_lag # this are used downstream in legends, useful.
from Figures.Figure_1.Figure_1_pointplot_lag import p_value_ttest_lag_ref_medial, p_value_ttest_lag_ref_lateral, p_value_ttest_common, p_value_ttest_strong
from Figures.Figure_1.Figure_1_violinplot_quartiles import pvalues, violinplot_data
fig1_p_value_violin = np.round(pvalues[0], 23)
fig1_means_violinplot = np.round(violinplot_data.groupby("Correlation quartiles").mean().values, 2)
fig1_sem_violinplot = np.round((violinplot_data.groupby("Correlation quartiles").std()/np.sqrt(violinplot_data.shape[0]/2)).values, 2)
with open(f"{output_folder_figures_calculations}/temp_data_figure_1.pkl", "rb") as fp:  # Unpickling
    sessions, high_distance, low_distance, ripples_lags, ripples_lags_inverted_reference, ripples_calcs, summary_corrs, distance_tabs = dill.load(fp)

clip = (-50, 50)
ripples_lags_clipped = ripples_lags[ripples_lags["Lag (ms)"].between(clip[0], clip[1])]

_ = summary_corrs[(summary_corrs["Comparison"]=="CA1-CA1")]

y =_["Correlation"]
x = _["Distance (µm)"]
# fit = np.polyfit(x, y, deg=1)
# predict = np.poly1d(fit)

# plt.plot( x, predict(x),color="k", alpha=0.5, linewidth=1)
# res = scipy.stats.linregress( x, y)

r, _ = pearsonr(x, y)
r_corr_distance_power_CA1_CA1 = round(r**2, 4)


fig1_n_high_dist = ripples_lags_clipped[ripples_lags_clipped["Type"] =="High distance (µm)"].shape[0]
fig1_n_animals_high_dist = ripples_lags_clipped[ripples_lags_clipped["Type"] =="High distance (µm)"]["Session"].unique().shape[0]
fig1_n_animals_low_dist = ripples_lags_clipped[ripples_lags_clipped["Type"] =="Low distance (µm)"]["Session"].unique().shape[0]
fig1_n_low_dist = ripples_lags_clipped[ripples_lags_clipped["Type"] =="Low distance (µm)"].shape[0]
fig1_mean_lagplot_summary = np.round(ripples_lags_clipped.groupby(["Type", "Session"])["Lag (ms)"].mean().groupby("Type").mean().values, 2)
fig1_sem_lagplot_summary = np.round(ripples_lags_clipped.groupby(["Type", "Session"])["Lag (ms)"].mean().groupby("Type").sem().values, 2)
fig1_mean_abs_lagplot_summary = np.round(ripples_lags_clipped.groupby(["Type", "Session"])["Absolute lag (ms)"].mean().groupby("Type").mean().values, 2)
fig1_sem_abs_lagplot_summary = np.round(ripples_lags_clipped.groupby(["Type", "Session"])["Absolute lag (ms)"].mean().groupby("Type").sem().values, 2)

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

# variables for figure 2
with open(f'{output_folder_figures_calculations}/temp_data_figure_2.pkl', 'rb') as f:
    session_id_fig2, session_trajs, columns_to_keep, ripples, real_ripple_summary,\
    lfp_per_probe, ripple_cluster_strong, ripple_cluster_weak, example_session = dill.load(f)

with open(f'{output_folder_calculations}/trajectories_by_strength.pkl', 'rb') as f:
    trajectories_by_strength = dill.load(f)

n_sessions_fig2 = trajectories_by_strength["Session"].unique().shape[0]

# variables for figure 3
with open(f'{output_folder_calculations}/ripples_features_sessions_across_all_ML.pkl', 'rb') as f:
    out = dill.load(f)

seed_ripples_by_hip_section_summary_common = pd.concat([q[3] for q in out])
seed_ripples_by_hip_section_summary_strong = pd.concat([q[6] for q in out])
seed_ripples_by_hip_section_summary_common= seed_ripples_by_hip_section_summary_common.reset_index().rename(columns={'index': 'Location seed'})
seed_ripples_by_hip_section_summary_strong= seed_ripples_by_hip_section_summary_strong.reset_index().rename(columns={'index': 'Location seed'})

data = seed_ripples_by_hip_section_summary_common
sem_common = data.groupby(["Reference", "Location seed"])["Percentage seed (%)"].sem().reset_index().pivot(columns="Reference", index="Location seed", values="Percentage seed (%)")
fig_3_sem_common = sem_common[["Medial","Central","Lateral"]]
mean_common = data.groupby(["Reference", "Location seed"])["Percentage seed (%)"].mean().reset_index().pivot(columns="Reference", index="Location seed", values="Percentage seed (%)")
fig_3_mean_common = mean_common[["Medial","Central","Lateral"]]

data = seed_ripples_by_hip_section_summary_strong
sem_strong = data.groupby(["Reference", "Location seed"])["Percentage seed (%)"].sem().reset_index().pivot(columns="Reference", index="Location seed", values="Percentage seed (%)")
fig_3_sem_strong = sem_strong[["Medial", "Central", "Lateral"]]
mean_strong = data.groupby(["Reference", "Location seed"])["Percentage seed (%)"].mean().reset_index().pivot(columns="Reference", index="Location seed", values="Percentage seed (%)")
fig_3_mean_strong = mean_strong[["Medial", "Central", "Lateral"]]


# variables for figure 4
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

session_id_fig4 = session_id_example

with open(f"{output_folder_calculations}/spike_hists_summary_per_parent_area_by_seed_location.pkl", 'rb') as f:
    spike_hists = dill.load(f)

fig_4_summary_fraction_active_clusters_per_ripples_early = summary_fraction_active_clusters_per_ripples_early.groupby(["Session id", "Location seed"]).mean().reset_index()

fig_4_summary_fraction_active_clusters_per_ripples_late = summary_fraction_active_clusters_per_ripples_late.groupby(["Session id", "Location seed"]).mean().reset_index()

from Figures.Figure_4.Figure_4_clusters_per_ripple_early_late import ttest_early_clus_per_ripple, ttest_late_clus_per_ripple
fig_4_ttest_early_fraction = ttest_early_clus_per_ripple
fig_4_ttest_late_fraction = ttest_late_clus_per_ripple

from Figures.Figure_4.Figure_4_spiking_rate_early_late import ttest_early_spiking, ttest_late_spiking
fig_4_ttest_early_spiking = ttest_early_spiking
fig_4_ttest_late_spiking = ttest_late_spiking


fig_4_summary_spiking_early = pd.melt(tot_summary_early[["Lateral seed", "Medial seed", "Session id"]],
               value_vars=["Lateral seed", "Medial seed"],
               id_vars=["Session id"], var_name='Location seed', value_name='Spiking rate per 10 ms')\
    .groupby(["Session id", "Location seed"]).mean().reset_index()


fig_4_summary_spiking_late = pd.melt(tot_summary_late[["Lateral seed", "Medial seed", "Session id"]],
               value_vars=["Lateral seed", "Medial seed"], id_vars=["Session id"], var_name='Location seed',
               value_name='Spiking rate per 10 ms').groupby(["Session id", "Location seed"]).mean().reset_index()






