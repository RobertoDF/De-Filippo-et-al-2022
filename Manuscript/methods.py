from Utils.Settings import window_spike_clu, window_spike_hist, output_folder_calculations, neuropixel_dataset, output_folder_figures_calculations, var_thr
import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import dill
from tqdm import tqdm
import pandas as pd
from Utils.Settings import minimum_ripples_count_generated_in_lateral_or_medial_spike_analysis, neuropixel_dataset, lowcut, highcut, fs_lfp, ripple_dur_lim, min_ripple_distance, ripple_power_sp_thr, ripple_thr, start_w, stop_w, thr_rip_cluster
from tqdm import tqdm
import numpy as np

manifest_path = os.path.join(neuropixel_dataset, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)


sessions = cache.get_session_table()

sessions_ids_no_CA1 = []
for idx, data in sessions.iterrows():
    if "CA1" not in data["ecephys_structure_acronyms"]:
        sessions_ids_no_CA1.append(idx)



manifest_path = f"{neuropixel_dataset}/manifest.json"
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()

out = []
for session_id in tqdm(sessions.index):
    try:
        with open(f'/alzheimer/Roberto/Allen_Institute/Processed_lfps/lfp_errors_{session_id}.pkl', 'rb') as f:
            errors, noise, time_non_increasing = dill.load(f)
            out.append([errors, noise, time_non_increasing])
    except:
        pass

time_is_monotonic = pd.DataFrame([e for q in out for e in q[2]], columns=["probe_id", "non monotonic"])
probes = cache.get_probes()
session_id_non_monotonic = probes.loc[time_is_monotonic[time_is_monotonic["non monotonic"]==True]["probe_id"]]["ecephys_session_id"].values[0]

session_id = pd.Series([q[0][0] for q in out], name="session_id")
mistakes = pd.Series([True if len(q[0][1]) > 0 else False for q in out], name="mistakes present")
lfp_identity = pd.concat([session_id, mistakes], axis=1)



with open(f"{output_folder_figures_calculations}/temp_data_figure_1.pkl", "rb") as fp:  # Unpickling
    sessions, high_distance, low_distance, ripples_lags, ripples_lags_inverted_reference, ripples_calcs, summary_corrs = dill.load(fp)

quartiles_distance = summary_corrs[summary_corrs["Comparison"] == "CA1-CA1"]["Distance (µm)"].quantile(
    [0.25, 0.5, 0.75])

methods = {"Dataset": f"Our analysis was based on the Visual Coding - Neuropixels dataset provided by the Allen Institute and available at "
                      f"https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html. "
                      f"We excluded one session (session id = {session_id_non_monotonic}) because of an artifact in the LFP time series (time was not monotonically increasing). "
                      f"Two other sessions (session ids = {', '.join(map(str, lfp_identity[lfp_identity['mistakes present'] == True]['session_id'].values))})"
                      f" were excluded because of duplicated LFP traces. Other {len(sessions_ids_no_CA1)} sessions were excluded because of absence of "
                      f"recording electrodes in CA1 (session ids={', '.join(map(str, sessions_ids_no_CA1))}). Our analysis was therefore focused on"
                      f" {len(ripples_calcs)} sessions, average animal age = "
                      f"{round(sessions.loc[ripples_calcs.keys()]['age_in_days'].mean(),2)} ± "
                      f"{round(sessions.loc[ripples_calcs.keys()]['age_in_days'].sem(),2)}. Sex: males n = {sessions.loc[ripples_calcs.keys()]['sex'].value_counts().loc['M']}, "
                      f"females n = {sessions.loc[ripples_calcs.keys()]['sex'].value_counts().loc['F']}. Genotypes: wt/wt  n = {sessions.loc[ripples_calcs.keys()]['full_genotype'].value_counts().loc['wt/wt']}, "
                      f"Sst-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt n = {sessions.loc[ripples_calcs.keys()]['full_genotype'].value_counts().loc['Sst-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt']}, "
                      f"Vip-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt n = {sessions.loc[ripples_calcs.keys()]['full_genotype'].value_counts().loc['Vip-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt']}, "
                      f"Pvalb-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt n = {sessions.loc[ripples_calcs.keys()]['full_genotype'].value_counts().loc['Pvalb-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt']}. "
                      f"Average probe count per session = {round(sessions.loc[ripples_calcs.keys()]['probe_count'].mean(), 2)} ± "
                      f"{round(sessions.loc[ripples_calcs.keys()]['probe_count'].sem(), 2)}. "
                      f"Average number of recording channels per session = {round(sessions.loc[ripples_calcs.keys()]['channel_count'].mean(), 2)} ± "
                      f"{round(sessions.loc[ripples_calcs.keys()]['channel_count'].sem(), 2)}. Probes in each session were numbered according to the position "
                      f"on the M-L axis, with probe number 0 being the most medial. Channels with ambiguous area annotations were discarded (e.g. HPF instead of CA1). "
                      f"We found a number of of small artifacts in a variety of sessions, all this timepoints were excluded from the analysis (for more informations: "
                      f"https://github.com/RobertoDF/Allen_visual_dataset_artifacts). Further details about data acquisition can be found at "
                      f"https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/80/75/8075a100-ca64-429a-b39a-569121b612b2/neuropixels_visual_coding_-_white_paper_v10.pdf. "
                      "Visualization of recording locations was performed with brainrender {Claudi, 2021 #1134}.",

        "Ripples detection": f"The LFP traces sampled at {fs_lfp} Hz were filtered using a 6th order Butterworth bandpass filter between {lowcut} and {highcut}. "
                           f"Ripples were detected on CA1 LFP traces, the best channel (higher ripple strength) was selected by looking at the SD of the envelope of the filtered trace,"
                           f" if multiple SD peaks were present across space (possibly caused by sharp waves in stratum radiatum and ripple activity in stratum pyramidale)"
                           f" we subsequently looked at the channel with higher skewness, in this way we could reliably identify the best ripple channel. "
                           f"The envelope of the filtered trace was calculated using the Hilbert transform (scipy.signal.hilbert). Ripple threshold was set at {ripple_thr} SDs. "
                           f"Start and stop times were calculated using a 2 SDs threshold "
                           f"on the smoothed envelope with window = 5 (pandas.DataFrame.rolling) to account for ripple phase distortions. "
                           f"Ripple duration was limited at >{ripple_dur_lim[0]} s and <{ripple_dur_lim[1]} s. "
                           f"Candidate ripples were excluded if preceded by another ripple in "
                           f"a window of {min_ripple_distance} s. We estimated power density of each candidate using a "
                           f"periodogram with constant detrending (scipy.signal.periodogram) on the raw LFP trace, "
                           f"we checked the presence of a peak >"
                           f" {ripple_power_sp_thr} Hz, candidates not fulfilling this condition were discarded, this condition was meant to "
                             f"reduce the number of detected false positives. Ripple candidates  detected during running epochs "
                             f"were discarded, an animal was considered to be running if his standardized speed was higher than the 10th percentile plus 0.06. "
                             f"Candidates were also discarded if no behavioral data was available. Code for the detection of ripples resides in 'Calculate_ripples.py'. ",

        "Correlation and lag analysis": "In each session we uniquely used ripples from the CA1 channel with the strongest ripple activity, "
                                        f"we looked at the LFP activity in all brain areas recorded in a window of {start_w*1000} ms pre ripple start and {stop_w*1000} ms post ripple start, this "
                                        f"broad windows account for possible travelling delays due to distance. "
                                        f"For each brain area we picked the channel with higher SD of the envelope of the filtered trace. "
                                        f"For each ripple considered we calculated integral of the envelope of the filtered trace (∫Ripple) and the integral of the raw LFP (ripple-induced voltage deflection, RIVD). "
                                        f"After discarding channels with weak ripple activity (envelope variance < {var_thr}), we computed the pairwise pearson correlation of the envelope traces of CA1 channels (pandas.DataFrame.corr). "
                                        f"For the lag analysis we first identified pairs of CA1 that satisfied a distance requirements. Distance threshold were set at 25% ({round(quartiles_distance[0.25], 2)} µm) "
                                        f"and 75% ({round(quartiles_distance[0.75], 2)} µm) of the totality of distances. "
                                        f"For each ripple detected in the reference channel we identifired the nearest neighbour in the other channel. The analysis was repeated after dividing ripples"
                                        f" in strong (top 10% ∫Ripple) and common ripples (all remaining ripples) per session. Code for the correlation and lag analysis resides in 'Calculations_Figure_1.py'.",

        "Ripple spatio-temporal propagation maps and ripple seed analysis": "The hippocampus was divided in three section with equal number of recordings. "
                                               f"Channels with weak ripple activity (envelope variance < {var_thr}) were discarded. Sessions with recording locations only in one hippocampal sections "
                                               f"or with less than 1000 ripples in the channel with strongest ripple activity were discarded as well. For each ripple detected on the reference CA1 "
                                                f"channel we identified ripples in other CA1 channels happening in a window of ± {thr_rip_cluster*1000} ms, "
                                               f"this events were grouped together in a 'cluster'. If more than one event was detected on the same probe we kept only the first event. 'Clusters' were subsequently "
                                               f"divided according to ∫Ripple on the reference electrode in strong and common ripples. Lag maps were result of averaging lags for each probe. "
                                                                        f"Code for the calculations of propagation maps resides in 'Calculate_trajectories.py'.  ",
        "Ripple-associated spiking activity": f"We focused on sessions with clear ripple activity (envelope variance > {var_thr}) in all three hippocampal sections and at least {minimum_ripples_count_generated_in_lateral_or_medial_spike_analysis} "
                                              f"ripples generated both medially and laterally. The reference was always placed in the central section, here it was possible"
                                              f" to identify ripples generated medially and laterally. We only considered ripples that were detected in at least half of the recording electrodes"
                                              ' (in the code: "spatial engagment" > 0.5). For each ripple we computed a histogram of spiking activity of regions belonging to the hippocampal formation (HPF) in a '
                                              f'window of {np.sum(window_spike_hist)} s '
                                              f"centered on the ripple start in each probe. We averaged all the computed histograms to create a spatial profile of spiking activity. "
                                              f" To compare spiking activity between sessions we interpolated (xarray.DataArray.interp) the difference between medial ripples-induced spiking and"
                                              f" lateral ripples-induced spiking over space (this was necessary because probes in each sessions have different M-L coordinates) and time. We calculated the number of active cells (at least one spike) "
                                              f"and spiking rate of each cluster per ripple in a window of {window_spike_clu[1]} s starting from ripple start. We repeated the analysis separating "
                                              f"the 0-50 ms and 50-120 ms post ripple start windows."
           }
