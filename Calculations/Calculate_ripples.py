from Utils.Utils import calculations_per_ripple, clean_ripples_calculations, select_quiet_part, ripple_finder, butter_bandpass_filter
from tqdm import tqdm
import dill
import numpy as np
from scipy.signal import hilbert
import pandas as pd
import pickle
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from multiprocessing import Pool
from time import perf_counter
import os
from Utils.Settings import output_folder_calculations, neuropixel_dataset,  lowcut, highcut, fs_lfp, start_w, stop_w


t1_start = perf_counter()

manifest_path = f"{neuropixel_dataset}/manifest.json"

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
sessions = cache.get_session_table()

out = {}

loop_over = range(sessions.shape[0])

# select session
for session_n in tqdm(loop_over):
    session_id = sessions.index.values[session_n]
    print(f"session number: {session_n}, {session_id}")
    session_id = sessions.index.values[session_n]
    session = cache.get_session_data(session_id)  #, amplitude_cutoff_maximum = np.inf, presence_ratio_minimum = -np.inf, isi_violations_maximum = np.inf)

    if os.path.exists(f'/alzheimer/Roberto/Allen_Institute/Processed_lfps/lfp_errors_{session_id}.pkl') == True:
        with open(f'/alzheimer/Roberto/Allen_Institute/Processed_lfps/lfp_per_probe_{session_id}.pkl', 'rb') as f:
            lfp_per_probe = dill.load(f)

        with open(f'/alzheimer/Roberto/Allen_Institute/Processed_lfps/behavior_{session_id}.pkl', 'rb') as f:
            behavior, start_running, stop_running, start_quiet, stop_quiet = dill.load(f)

        with open(f'/alzheimer/Roberto/Allen_Institute/Processed_lfps/lfp_errors_{session_id}.pkl', 'rb') as f:
            errors, noise, time_non_increasing = dill.load(f)

        if len(errors[1]) == 0:


            if "CA1" in sessions.iloc[session_n]["ecephys_structure_acronyms"]:
                if len(lfp_per_probe) != 0:

                    ts = []
                    which_channel = []
                    for probe_n, selected_lfp in tqdm(enumerate(lfp_per_probe)):
                        if "CA1" in selected_lfp.area:

                            sig = selected_lfp.sel(area="CA1")
                            sig = select_quiet_part(sig, start_quiet, stop_quiet)

                            filtered = butter_bandpass_filter(np.nan_to_num(sig.values), lowcut, highcut, fs_lfp, order=6)
                            analytic_signal = hilbert(filtered)
                            amplitude_envelope = np.abs(analytic_signal)
                            threshold_ripples = np.std(amplitude_envelope) * 5
                            ts.append(threshold_ripples)
                        else:
                            threshold_ripples = np.nan
                            ts.append(threshold_ripples)

                        print(f"threshold at probe {probe_n}:  {threshold_ripples}")

                    fs = fs_lfp
                    probe_selected_ripples = np.nanargmax(ts)
                    threshold_ripples = np.nanmean(ts)

                    print("probe_selected: ", probe_selected_ripples)

                    input_rip = []
                    for probe_n, lfp in enumerate(lfp_per_probe):
                        for area in lfp.area.values:
                            if area in ["CA1"]: #, "SUB", "ProS", "CA3", "CA2", "DG"]:
                                print(f"Extract ripples from {probe_n}-{area}")
                                sig = lfp.sel(area=area)
                                input_rip.append((sig, fs, threshold_ripples, probe_n, area))

                    with Pool(processes=len(input_rip)) as pool:
                        r = pool.starmap_async(ripple_finder, input_rip)
                        list_ripples = r.get(timeout=60 * 30)
                        pool.terminate()

                    print("ripples detection completed")

                    ripples = pd.concat(list_ripples)
                    ripples.reset_index(drop=True, inplace=True)

                    ripples_subselect = ripples[["Start (s)", "Stop (s)", "Amplitude (mV)", "Probe number-area"]]

                    sliced_lfp_per_probe = []
                    ripples_start = ripples_subselect[ripples_subselect["Probe number-area"] == f"{probe_selected_ripples}-CA1"][
                        "Start (s)"].values
                    ripples_stop = ripples_subselect[ripples_subselect["Probe number-area"] == f"{probe_selected_ripples}-CA1"][
                        "Stop (s)"].values


                    length = int(np.floor((start_w + stop_w) * fs))-1 # subtract one sample, for unclear reason this is necessary to make the xarrays same length, otherwise one is always one smaple shorter

                    for start in tqdm(ripples_start):
                        res = []
                        for probe_n, lfp in enumerate(lfp_per_probe):
                            _ = lfp.sel(time=slice(start - start_w, start + stop_w))
                            res.append(_[:length, :])
                        sliced_lfp_per_probe.append(res)

                    input_corr = []

                    for start, stop, lfp in zip(ripples_start, ripples_stop, sliced_lfp_per_probe):
                        input_corr.append((start, stop, lfp, 120, 250, fs_lfp, length))

                    with Pool(processes=30) as pool:
                        r = pool.starmap_async(calculations_per_ripple, input_corr, chunksize=80)
                        res_calc_per_ripple = r.get()
                        pool.close()

                    ripple_power = pd.DataFrame([q[0].loc["Ripple area (mV*s)"] for q in res_calc_per_ripple], index=[q[1] for q in res_calc_per_ripple])
                    ripple_power.index.name = "Time (s)"
                    ripple_power.name = "∫Ripple"
                    ripple_power.columns.names = ['Probe number', "Area"]
                    ripple_power.index = ripple_power.index.astype(int)

                    pos_area = pd.DataFrame([q[0].loc["Positive area (mV*s)"] for q in res_calc_per_ripple], index=[q[1] for q in res_calc_per_ripple])

                    pos_area.index.name = "Time (s)"
                    pos_area.name = "RIVD"
                    pos_area.columns.names = ['Probe number', 'Area']
                    pos_area.index = pos_area.index.astype(int)
                    neg_area = pd.DataFrame([q[0].loc["Negative area (mV*s)"] for q in res_calc_per_ripple], index=[q[1] for q in res_calc_per_ripple])

                    neg_area.index.name = "Time (s)"
                    neg_area.name = "RIVD"
                    neg_area.columns.names = ['Probe number', 'Area']
                    neg_area.index = neg_area.index.astype(int)

                    output_calc = [ripple_power, pos_area, neg_area]

                    probe_ns = []
                    area = []
                    dv = []
                    ap = []
                    lr = []
                    for probe_n, lfp in enumerate(lfp_per_probe):
                        probe_ns.extend([probe_n] * lfp.shape[1])
                        area.extend(lfp.area.values)
                        dv.extend(lfp.dorsal_ventral_ccf_coordinate.values)
                        ap.extend(lfp.anterior_posterior_ccf_coordinate.values)
                        lr.extend(lfp.left_right_ccf_coordinate.values)

                    spatial_info = pd.DataFrame([probe_ns, area, dv, ap, lr], index=["Probe number", "Area", "D-V (µm)", "A-P (µm)", "L-R (µm)"]).T.infer_objects()

                    # add positions to ripples table
                    print("add positions to ripples table")
                    out_t = []
                    for row in tqdm(ripples[["Probe number", "Area"]].iterrows()):
                        probe_n = row[1]["Probe number"]
                        area = row[1]["Area"]
                        out_t.append(
                            spatial_info[(spatial_info["Probe number"] == probe_n) & (spatial_info["Area"] == area)][
                                ["D-V (µm)", "A-P (µm)", "L-R (µm)"]])
                    ripples = pd.concat([ripples, pd.concat(out_t).reset_index(drop=True)], axis=1)

                    out[session_id] = [output_calc, spatial_info, session.probes.index.values, ripples, [behavior, start_running, stop_running, start_quiet, stop_quiet], probe_selected_ripples]
                else:
                    print("no LFP available")
            else:
                print("no CA1 present>>>skip session")
        else:
            print("LFP with errors")
    else:
        print(f"File session {session_id} does not exists")
t1_stop = perf_counter()

print("Elapsed time during the whole program in seconds:",
      t1_stop - t1_start)

ripples_calcs = clean_ripples_calculations(out)
with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'wb') as f:
    dill.dump(ripples_calcs, f)


