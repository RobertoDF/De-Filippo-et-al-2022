import os
from time import perf_counter
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from tqdm import tqdm
from Utils.Process_lfp import *
from multiprocessing.pool import ThreadPool, Pool
import dill
from Utils.Settings import output_folder_processed_lfps, neuropixel_dataset


# run this with python 3.8, 3.7 has a problem pickling multiprocessing module.
# 1. Create environment: `conda create --name process_lfp python=3.8`
# 2. Activate environment: 'conda activate process_lfp'
# 3. Install requirements: `pip install --no-deps -r requirements_process_lfp.txt`


t1_start = perf_counter()

manifest_path = f"{neuropixel_dataset}/manifest.json"

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()

loop_over = range(sessions.shape[0])

# select session
for session_n in tqdm(loop_over):
    print(perf_counter())
    session_id = sessions.index.values[session_n]
    print(f"session number: {session_n}, {session_id}")
    session = cache.get_session_data(session_id)  #, amplitude_cutoff_maximum = np.inf, presence_ratio_minimum = -np.inf, isi_violations_maximum = np.inf)

    behavior, start_running, stop_running, start_quiet, stop_quiet = process_behavior(session)
    #  TODO:  divide quiet in alert and relaxed

    downsampling_factor = 1
    fs_lfp = 1250 / downsampling_factor
    print("LFP sampled at", fs_lfp, "Hz")

    channels_table = cache.get_channels()
    probes = cache.get_probes()
    # sorting L-R by ca1
    probe_ids = session.channels.groupby("probe_id").mean().sort_values("left_right_ccf_coordinate").index
    probe_ids = probe_ids[probes.loc[probe_ids]["has_lfp_data"]]

    if len(probe_ids) > 0:

        input_extract_lfp = [(session, probe_id, channels_table, downsampling_factor) for probe_id in probe_ids]

        output = []

        with ThreadPool(processes=len(probe_ids)) as pool:
            res = pool.starmap_async(extract_lfp_and_check_for_consistency, input_extract_lfp)
            output = res.get(timeout=60*20)
            pool.close()

        lfp_per_probe_all = [i[0] for i in output if isinstance(i[0], list) is False]  # filter empties out
        noise = [i[1] for i in output]
        time_non_increasing = [i[2] for i in output]

        lfp_per_probe_all = divide_cortex_in_d_s(lfp_per_probe_all)

        check_lfp = [q[:1000, :].data for q in lfp_per_probe_all]
        errors = check_lfp_equality(check_lfp, session_id)

        if len(lfp_per_probe_all) != 0:

            kind = "higher_std"  # "mean", "middle_channel" or "higher_std"
            input_fun = [(lfp, kind,  fs_lfp, session, cache, start_quiet, stop_quiet) for lfp in lfp_per_probe_all]

            lfp_per_probe = []

            with Pool(processes=len(lfp_per_probe_all)) as pool: # Pool raise error in python 3.7, in 3.8 it is fine
                r = pool.starmap_async(extract_lfp_per_area, input_fun)
                lfp_per_probe = r.get(timeout=60*45)
                pool.terminate()

            print(session.channels.groupby("probe_id").mean().sort_values("left_right_ccf_coordinate")[
                      "left_right_ccf_coordinate"])
            print([lfp.name for lfp in lfp_per_probe])

            print("Save LFP per area")
            with open(f"{output_folder_processed_lfps}/lfp_per_probe_{sessions.index.values[session_n]}.pkl",
                    "wb") as fp:
                dill.dump(lfp_per_probe, fp)

            with open(f"{output_folder_processed_lfps}/behavior_{sessions.index.values[session_n]}.pkl",
                      "wb") as fp:
                dill.dump([behavior, start_running, stop_running, start_quiet, stop_quiet], fp)


            with open(f"{output_folder_processed_lfps}/lfp_errors_{sessions.index.values[session_n]}.pkl",
                    "wb") as fp:
                dill.dump([errors, noise, time_non_increasing], fp)
        else:
            print("no LFP available")
    else:
        print("Probes do not have LFP data")

t1_stop = perf_counter()

print("Elapsed time during the whole program in seconds:",
      t1_stop - t1_start)


