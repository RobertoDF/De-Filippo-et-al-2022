import numpy as np
import pandas as pd
import pandas as pd
from Utils.Utils import acronym_to_main_area, acronym_to_graph_order, select_quiet_part, std_skew_on_filtered,  find_nearest
import xarray as xr
from scipy import signal
from itertools import groupby

def process_behavior(session):
    running_speed = session.running_speed
    running_speed = pd.Series(running_speed['velocity'].values, index=running_speed['start_time'] + (
            running_speed['end_time'] - running_speed['start_time']) / 2)
    try:
        pupil_data = session.get_pupil_data()
        pupil_size = 3.14 * ((pupil_data['pupil_height'] / 2) * (pupil_data['pupil_width'] / 2)) / 1000
        pupil_pos_x = pupil_data['pupil_center_x']
        pupil_pos_y = pupil_data['pupil_center_y']
        behavior = pd.concat([running_speed, pupil_size, pupil_pos_x, pupil_pos_y], axis=1).interpolate()
        behavior.columns = ['Running speed', 'Pupil size', "Pupil position x", "Pupil position y"]
        behavior["Pupil size"][behavior["Pupil size"] > 2] = 2
        behavior = behavior.loc[:running_speed.index[-1], :]
        print("pupil data present")
    except:
        print("no pupil data")
        behavior = pd.DataFrame(running_speed)
        behavior.columns = ['Running speed']


    behavior = (behavior - behavior.mean()) / behavior.std()
    behavior["Running speed"][behavior["Running speed"] > 5] = 5
    behavior["Running speed"][behavior["Running speed"] < -1] = -1

    behavior = behavior.rolling(20, center=True).mean().transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    behavior.index.name = 'time'
    # behavior.reset_index(inplace=True)

    temp = behavior.dropna()
    temp1 = temp[temp["Running speed"] > temp["Running speed"].quantile(0.1) + 0.06].index.values
    start = temp1[np.diff(temp1, prepend=0) > 0.1]
    stop = temp1[np.diff(temp1, append=temp1[-1] + 2) > 0.1]

    if stop[0] < start[0]:
        start = np.insert(start, 0, temp.index[0])
        stop = np.append(stop, temp.index[-1])

    if start[-1] > stop[-1]:
        stop = np.append(stop, temp.index[-1])

    mask = np.roll(np.roll(start, -1) - stop > 1, 1)
    mask[0] = True

    start_running = start[mask]

    mask = np.roll(start, -1) - stop > 1
    mask[-1] = True
    stop_running = stop[mask]

    mask = np.roll(start_running, -1) - stop_running > 20 #quiet at least 20 seconds long
    start_quiet = stop_running[mask] + 5

    mask = np.roll(np.roll(start_running, -1) - stop_running > 20, 1)
    stop_quiet = start_running[mask] - 5

    if stop_running[-1] < running_speed.index[-1]:
        start_quiet = np.append(start_quiet, stop_running[-1] + 5)
        stop_quiet = np.append(stop_quiet, running_speed.index[-1] - .1)

    if start_running[0] > running_speed.index[0]:
        start_quiet = np.insert(start_quiet, 0,  running_speed.index[0])
        stop_quiet = np.insert(stop_quiet, 0, start_running[0] - 5)

    start_ = start_running[stop_running - start_running > 5]
    stop_running = stop_running[stop_running - start_running > 5]
    start_running = start_

    start_ = start_quiet[stop_quiet - start_quiet > 5]
    stop_quiet = stop_quiet[stop_quiet - start_quiet > 5]
    start_quiet = start_

    if "Pupil size" not in behavior.columns:
        behavior["Pupil size"] = 0
    print("behavior processed")

    return behavior, start_running, stop_running, start_quiet, stop_quiet


def extract_lfp_per_area(lfp, kind,  fs_lfp, session, cache, start_quiet, stop_quiet):

    print(f"Processing probe id: {lfp.name} \n")

    channels_table = cache.get_channels()
    if kind == "mean":
        temp = lfp.groupby("area").mean("channel")
        order = np.argsort([acronym_to_graph_order(area) for area in temp.area.values])  # order by main structure
        temp = temp[:, order]
        dorsal_ventral_ccf_coordinate = session.channels.loc[lfp["channel"]]["dorsal_ventral_ccf_coordinate"]
        anterior_posterior_ccf_coordinate = session.channels.loc[lfp["channel"]]["anterior_posterior_ccf_coordinate"]
        left_right_ccf_coordinate = session.channels.loc[lfp["channel"]]["left_right_ccf_coordinate"]
        coord = pd.DataFrame([lfp.area.values, dorsal_ventral_ccf_coordinate, anterior_posterior_ccf_coordinate,
                              left_right_ccf_coordinate],
                             index=["area", "dorsal_ventral_ccf_coordinate", "anterior_posterior_ccf_coordinate"
                                 ,"left_right_ccf_coordinate"]).T
        coord_x_area = coord.groupby("area").mean()
        order = np.argsort([acronym_to_graph_order(area) for area in coord_x_area.index])  # order by main structure
        coord_x_area = coord_x_area.iloc[order]
        temp = temp.assign_coords(
            dorsal_ventral_ccf_coordinate=("area", coord_x_area["dorsal_ventral_ccf_coordinate"].values))
        temp = temp.assign_coords(
            anterior_posterior_ccf_coordinate=("area", coord_x_area["anterior_posterior_ccf_coordinate"].values))
        temp = temp.assign_coords(left_right_ccf_coordinate=("area", coord_x_area["left_right_ccf_coordinate"].values))

    elif kind == "middle_channel":
        temp = lfp.groupby("area").map(pick_middle_channel_vertical, args=channels_table)
        #TODO add coordinates

    elif kind == "higher_std":
        lfp_quiet = select_quiet_part(lfp, start_quiet, stop_quiet)

        res = xr.apply_ufunc(std_skew_on_filtered, lfp_quiet.chunk(
            {"time": lfp.shape[0], "channel": 1}), fs_lfp, input_core_dims=[["time"], []],
                           output_core_dims=[["measures"]],
                           output_dtypes=lfp.dtype, dask_gufunc_kwargs=dict(output_sizes={"measures": 2}),
                           vectorize=True, dask="parallelized").compute()

        lfp = lfp.assign_coords(
            std_filtered=("channel", res[:,0].values))
        lfp = lfp.assign_coords(
            skew_filtered=("channel", res[:,1].values))

        temp = lfp.groupby("area").map(pick_higher_std_skew)
        order = np.argsort([acronym_to_graph_order(area.split("-")[0]) for area in temp.area.values])  # order by main structure
        temp = temp[:, order]

        temp = temp.assign_coords(dorsal_ventral_ccf_coordinate=(
        "area", session.channels.loc[temp["channel"]]["dorsal_ventral_ccf_coordinate"].values))
        temp = temp.assign_coords(anterior_posterior_ccf_coordinate=(
        "area", session.channels.loc[temp["channel"]]["anterior_posterior_ccf_coordinate"].values))
        temp = temp.assign_coords(left_right_ccf_coordinate=(
        "area", session.channels.loc[temp["channel"]]["left_right_ccf_coordinate"].values))

    return temp


def extract_lfp_and_check_for_consistency(session, probe_id, channels_table, downsampling_factor):
    """ name dataframe = probe_id
    extract lfp and check for inconsinstencies: time has to monotonically increase, find weird artifacts """

    print(f"Processing probe {probe_id}. ")

    lfp = session.get_lfp(probe_id)
    time = lfp.time.values

    if np.all(time[1:] >= time[:-1]): # avoid when time is not linear

        lfp = lfp[:, np.in1d(lfp['channel'].values, channels_table.index.values)]  # some channels are missing because noisy(??)
        areas = session.channels.loc[lfp["channel"]]["ecephys_structure_acronym"]
        vertical_position = session.channels.loc[lfp["channel"]]["probe_vertical_position"]
        lfp = lfp.assign_coords(area=("channel", areas))
        lfp = lfp.assign_coords(vertical_position=("channel", vertical_position))
        lfp = lfp[::downsampling_factor, ~pd.isna(areas)]  # downsampling
        lfp = lfp * 1000 # mV
        lfp.name = str(probe_id)
        # clean from invalid_times
        invalid = session.invalid_times
        for q in range(invalid.shape[0]):
            if ("all_probes" in invalid.loc[q, "tags"]) or \
                    (session.probes.loc[probe_id]["description"] in invalid.loc[q, "tags"]):
                print("removing noise from " + session.probes.loc[probe_id]["description"])
                sta = find_nearest(lfp.time.values, invalid.loc[q, "start_time"])
                sto = find_nearest(lfp.time.values, invalid.loc[q, "stop_time"])
                lfp.loc[sta:sto] = np.nan

        noise = find_weird_flat_LFP(lfp)
        print(f"Additional noise probe {probe_id} found at:")
        print(f"start at {[q[0] for q in noise]} s")
        print(f"stop at {[q[1] for q in noise]} s")
        for range_noise in noise:
            lfp.loc[range_noise[0]-0.5:range_noise[1]+0.5] = np.nan

        noise = (probe_id, noise)
        #lfp.to_netcdf(f'/alzheimer/Roberto/Allen_Institute/temp/probe_{probe_id}.nc')

        time_non_increasing = (probe_id, False)

    else:
        print(f"weird problem with LFP probe {probe_id}, times non consecutive?!?!")
        lfp = []
        noise = (probe_id, [])
        time_non_increasing = (probe_id, True)

    # with open( f"/alzheimer/Roberto/Allen_Institute/Processed_lfps/lfp_errors_{probe_id}.pkl",
    #         "wb") as fp:
    #     dill.dump([noise, time_non_increasing], fp)

    return lfp, noise, time_non_increasing


def divide_cortex_in_d_s(lfp_per_probe_all):
    out = []
    for lfp in lfp_per_probe_all:
        parent_area = []
        for area in lfp.area.values:
            parent_area.append(acronym_to_main_area(area))
        lfp = lfp.assign_coords(
                parent_area=("channel", parent_area))
        sup_number = np.sum(lfp.parent_area=="Isocortex").values//2
        deep_number = np.sum(lfp.parent_area=="Isocortex").values - np.sum(lfp.parent_area == "Isocortex").values//2
        lfp = lfp.rename({"area": "area_orig"})
        lfp["area"] = lfp["area_orig"].copy()
        lfp["area"][-sup_number:] = lfp["area"][-sup_number:].values + "-s"
        lfp["area"][-deep_number-sup_number:-sup_number] = lfp["area"][-deep_number-sup_number:-sup_number].values + "-d"

        out.append(lfp)

    return out

def pick_middle_channel_vertical(sel, channels_table):
    x = channels_table.loc[sel["channel"], "probe_vertical_position"].to_numpy()
    x = x - np.min(x)
    idx = np.abs(x - np.max(x) / 2).argmin()
    return sel[:, idx]


def pick_higher_std_skew(sel):
    #check peaks: we want one in CA1
    peaks_std, _ = signal.find_peaks(sel.std_filtered.values, height=np.round(np.mean(sel.std_filtered).values,4),
                                 prominence=np.round(np.std(sel.std_filtered).values,4))
    if len(peaks_std) <= 1:
        out = sel[:, sel.std_filtered.argmax()]
    else:
        print(f"double peak in {np.unique(sel.area)}")
        idx = min(peaks_std, key=lambda x: abs(x - sel.skew_filtered.argmax()))# if there are two peaks I pick the one with higher skewness
        out = sel[:, idx]
    return out

def find_weird_flat_LFP(lfp):

    # x = np.diff(np.var(lfp, axis=1))
    x = np.diff(np.std(lfp, axis=1)) ** 2
    # x = (x < 2e-4) & (x > -2e-4)
    x = x < 4e-06
    ranges = [list(g) for k, g in groupby(range(len(x)), lambda idx: x[idx]) if k == 1]
    time = lfp.time.values
    final = [[time[r[0]], time[r[-1]]] for r in ranges if len(r) > 20]

    return final

def check_lfp_equality(out, session_id):

    to_check = np.concatenate(out, axis=1)
    errors = []
    for n, column in enumerate(to_check.T):
        if np.array(range(to_check.shape[1]))[np.sum(np.equal(column, to_check.T), axis=1) > 999].shape[0] > 1:
            errors.append([n, np.array(range(to_check.shape[1]))[np.sum(np.equal(column, to_check.T), axis=1) > 999]])
    if len(errors) > 0:
        print (f"Lfp errors in session {session_id}")
    else:
        print(f"All fine in session {session_id}")
    errors = (session_id, errors)
    return errors


