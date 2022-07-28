import dill
from sklearn.preprocessing import MinMaxScaler
from Utils.Utils import clean_ripples_calculations, find_ripples_clusters_new, \
    get_trajectory_across_time_space_by_strength
from scipy.stats import zscore
import pandas as pd
from Utils.Settings import var_thr, root_github_repo, output_folder_figures_calculations, output_folder_calculations, output_folder_processed_lfps

with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)


# calculate session example
session_id = 768515987

with open(f'{output_folder_processed_lfps}/lfp_per_probe_{session_id}.pkl', 'rb') as f:
    lfp_per_probe = dill.load(f)

ripples = ripples_calcs[session_id][3].copy()

ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)

ripples = ripples.sort_values(by="Start (s)").reset_index(drop=True)

ripples = ripples[ripples["Area"] == "CA1"]

ripples = ripples.reset_index().rename(columns={'index': 'Ripple number'})

try:
    ripples["Z-scored ∫Ripple"] = ripples.groupby("Probe number-area").apply(lambda group:  zscore(group["∫Ripple"], ddof=1)).droplevel(0)
except:
    ripples["Z-scored ∫Ripple"] = ripples.groupby("Probe number-area").apply(lambda group:  zscore(group["∫Ripple"], ddof=1)).T

ripples["Local strong"] = ripples.groupby("Probe number").apply(lambda x: x["∫Ripple"] > x["∫Ripple"].quantile(.9) ).sort_index(level=1).values

spatial_info = ripples_calcs[session_id][1].copy()
position = "Medial"
source_area = str(ripples["Probe number"].min()) + "-CA1"

session_trajs = get_trajectory_across_time_space_by_strength(session_id, ripples, spatial_info, source_area, position)

##### I want shaded error bars, thats's why I do this "manually", this function is used under the hood in "get_trajectory_across_time_space"
real_ripple_summary = find_ripples_clusters_new(ripples, source_area)

columns_to_keep = real_ripple_summary.loc[:, real_ripple_summary.columns.str.contains('lag')].columns

## example

n = 332 #start at 2171
idxs_cluster = real_ripple_summary.loc[n][real_ripple_summary.loc[n].index.str.contains("ripple number")].sort_values()

ripple_cluster_strong = ripples.loc[idxs_cluster]

scaler = MinMaxScaler()
colors_idx = scaler.fit_transform(ripple_cluster_strong["M-L (µm)"].values.reshape(-1,1)) * 254
colors_idx = colors_idx.astype(int)
ripple_cluster_strong["color index"] = colors_idx
ripple_cluster_strong.sort_values(by="M-L (µm)", inplace=True)

n = 128  #start at 851.34
idxs_cluster = real_ripple_summary.loc[n][real_ripple_summary.loc[n].index.str.contains("ripple number")].sort_values().dropna()

ripple_cluster_weak = ripples.loc[idxs_cluster]
colors_idx = scaler.fit_transform(ripple_cluster_weak["M-L (µm)"].values.reshape(-1,1)) * 254
colors_idx = colors_idx.astype(int)
ripple_cluster_weak["color index"] = colors_idx
ripple_cluster_weak.sort_values(by="M-L (µm)", inplace=True)


columns_to_keep = real_ripple_summary.loc[:, real_ripple_summary.columns.str.contains('lag')].columns
proben_area = [q.split(" ")[0] for q in columns_to_keep]
probe_n = [q.split("-")[0] for q in proben_area]
area = [q.split("-")[1] for q in proben_area]

pos = []
for p, a in zip(probe_n, area):
    pos.append(spatial_info[(spatial_info["Probe number"] == int(p)) & (spatial_info["Area"] == a)]["M-L (µm)"])

lag_stronger = pd.concat([real_ripple_summary[
                                real_ripple_summary["∫Ripple"] > real_ripple_summary["∫Ripple"].quantile(.9)][
                                columns_to_keep].mean().reset_index(drop=True) * 1000,
                            pd.concat(pos).reset_index(drop=True)], axis=1)
lag_stronger.columns = ["Lag (ms)", "M-L (µm)"]
lag_stronger.sort_values(by="M-L (µm)", inplace=True)
lag_stronger["Session"] = session_id
lag_stronger["Probe number-area"] = proben_area
lag_stronger["Type"] = "Strong ripples"
lag_stronger["Location"] = position

lag_weaker = pd.concat([real_ripple_summary[
                              real_ripple_summary["∫Ripple"] < real_ripple_summary["∫Ripple"].quantile(.9)][
                              columns_to_keep].mean().reset_index(drop=True) * 1000,
                          pd.concat(pos).reset_index(drop=True)], axis=1)
lag_weaker.columns = ["Lag (ms)", "M-L (µm)"]
lag_weaker.sort_values(by="M-L (µm)", inplace=True)
lag_weaker["Session"] = session_id
lag_weaker["Probe number-area"] = proben_area
lag_weaker["Type"] = "Common ripples"
lag_weaker["Location"] = position

lag_tot = pd.concat([real_ripple_summary[
                              columns_to_keep].mean().reset_index(drop=True) * 1000,
                          pd.concat(pos).reset_index(drop=True)], axis=1)
lag_tot.columns = ["Lag (ms)", "M-L (µm)"]
lag_tot.sort_values(by="M-L (µm)", inplace=True)
lag_tot["Session"] = session_id
lag_tot["Probe number-area"] = proben_area
lag_tot["Type"] = "Total ripples"
lag_tot["Location"] = position


example_session = pd.concat([lag_stronger, lag_weaker, lag_tot]).reset_index(drop=True)


with open(f"{output_folder_figures_calculations}/temp_data_figure_2.pkl",
          "wb") as fp:
    dill.dump([session_id, session_trajs, columns_to_keep, ripples, real_ripple_summary,
               lfp_per_probe, ripple_cluster_strong, ripple_cluster_weak, example_session], fp)

# create brainrenders
exec(open(f"{root_github_repo}/Figures/Figure_2/Figure_2_brainrender.py").read())
exec(open(f"{root_github_repo}/Figures/Figure_2/Figure_2_brainrender_center.py").read())
exec(open(f"{root_github_repo}/Figures/Figure_2/Figure_2_brainrender_lateral.py").read())
exec(open(f"{root_github_repo}/Figures/Figure_2/Figure_2_brainrender_medial.py").read())


