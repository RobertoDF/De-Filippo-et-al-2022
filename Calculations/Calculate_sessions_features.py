import os

import dill
import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from tqdm import tqdm

from Utils.Settings import var_thr, output_folder_processed_lfps, output_folder_calculations, neuropixel_dataset
from Utils.Utils import acronym_to_main_area
from Utils.Utils import clean_ripples_calculations

manifest_path = os.path.join(neuropixel_dataset, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

with open('/alzheimer/Roberto/Allen_Institute/Processed_data/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)

sessions_durations = {}
for session_id in tqdm(list(ripples_calcs.keys())):
    with open(f'{output_folder_processed_lfps}/lfp_per_probe_{session_id}.pkl', 'rb') as f:
            lfp_per_probe = dill.load(f)

    session_duration = []
    for l in lfp_per_probe:
        session_duration.append(l.time.max() - l.time.min())
    sessions_durations[session_id] = np.mean(session_duration)

sessions_durations_quiet = {}
for session_id in tqdm(list(ripples_calcs.keys())):
    with open(f'{output_folder_processed_lfps}/behavior_{session_id}.pkl', 'rb') as f:
            behavior, start_running, stop_running, start_quiet, stop_quiet = dill.load(f)

    quiet_duration = []
    for start,stop in tqdm(zip(start_quiet, stop_quiet)):
        quiet_duration.append(stop-start)

    sessions_durations_quiet[session_id] = np.sum(quiet_duration)

sessions_durations_running = {}
for session_id in tqdm(list(ripples_calcs.keys())):
    with open(f'{output_folder_processed_lfps}/behavior_{session_id}.pkl', 'rb') as f:
            behavior, start_running, stop_running, start_quiet, stop_quiet = dill.load(f)

    running_duration = []
    for start,stop in tqdm(zip(start_running, stop_running)):
        running_duration.append(stop-start)

    sessions_durations_running[session_id] = np.sum(running_duration)


with open(f"{output_folder_calculations}/sessions_features.pkl", "wb") as fp:
    dill.dump([sessions_durations, sessions_durations_quiet, sessions_durations_running], fp)


out = {}

for session_id, sel in tqdm(ripples_calcs.items()):

    ripple_power = sel[0][0].copy()
    ripple_power = ripple_power.loc[:, ripple_power.var() > var_thr]
    out[session_id] = ripple_power.shape[0]

with open(f"{output_folder_calculations}/number_ripples_per_session_best_CA1_channel.pkl", "wb") as fp:
    dill.dump(out, fp)


sessions = cache.get_session_table()

out = []
out2 = []
for session_id in tqdm(sessions.index.values):
    session = cache.get_session_data(session_id,
                                 amplitude_cutoff_maximum = np.inf, presence_ratio_minimum = -np.inf, isi_violations_maximum = np.inf)  #, amplitude_cutoff_maximum = np.inf, presence_ratio_minimum = -np.inf, isi_violations_maximum = np.inf)

    units = session.units
    units["Session id"]=session_id
    out.append(units)
    session = cache.get_session_data(session_id)  #, amplitude_cutoff_maximum = np.inf, presence_ratio_minimum = -np.inf, isi_violations_maximum = np.inf)

    units = session.units
    units["Session id"]=session_id
    out2.append(units)

total_units = pd.concat(out)
total_clean_units = pd.concat(out)

with open(f"{output_folder_calculations}/total_units_table.pkl", "wb") as fp:
    dill.dump([total_units, total_clean_units], fp)
