import math
import random
from itertools import compress
from itertools import groupby
from operator import itemgetter
import dill as pickle
from Utils.Settings import neuropixel_dataset, utils_folder
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import scipy
import dill
from statannotations.Annotator import Annotator
from Utils.Settings import thr_start_stop, var_thr, output_folder_calculations, ripple_dur_lim, min_ripple_distance, ripple_power_sp_thr, ripple_thr, minimum_ripples_count_lag_analysis, thr_rip_cluster
from matplotlib.lines import Line2D
import pingouin as pg
import os
from Utils.Style import palette_ML, palette_HPF
from itertools import chain
from scipy.stats import zscore
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import animation
from matplotlib.colors import rgb2hex
from matplotlib.patches import Ellipse
from numpy import sum, count_nonzero
from scipy import signal
from scipy.signal import butter, sosfiltfilt
from scipy.signal import hilbert, medfilt, detrend
from scipy.stats import skew, pearsonr
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from Utils.Settings import lowcut, highcut, ripple_dur_lim

try:
    import tensorflow as tf
    from tensorflow import keras
    import sklearn
except:
    pass

try:
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    from allensdk.api.queries.ontologies_api import OntologiesApi
    import nrrd
    from allensdk.core.structure_tree import StructureTree
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    from allensdk.core.reference_space_cache import ReferenceSpaceCache, ReferenceSpace
    from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
    from allensdk.config.manifest import Manifest

    # define acronym map
    oapi = OntologiesApi()
    structure_graph = oapi.get_structures_with_sets([1])  # 1 is the id of the adult mouse structure graph
    # This removes some unused fields returned by the query
    structure_graph = StructureTree.clean_structures(structure_graph)
    tree = StructureTree(structure_graph)
    mcc = MouseConnectivityCache()
    structure_tree = mcc.get_structure_tree()
    # get the ids of all the structure sets in the tree
    structure_set_ids = structure_tree.get_structure_sets()

    summary_structures_all = pd.DataFrame(tree.nodes())
    summary_structures = pd.DataFrame(structure_tree.get_structures_by_set_id(
        [2]))  # Structures representing the major divisions of the mouse brain
    summary_structures = summary_structures.append(structure_tree.get_structures_by_acronym(["VS"]),
                                                   ignore_index=True) # append ventricular system
    summary_structures_finer = pd.DataFrame(structure_tree.get_structures_by_set_id(
        [3]))
    summary_structures_HP = pd.DataFrame(structure_tree.get_structures_by_set_id(
        [688152359]))


    # useful
    # pd.DataFrame(oapi.get_structure_sets(structure_set_ids))
    # pd.DataFrame(tree.nodes())

    acronym_color_map = tree.value_map(lambda x: x['acronym'], lambda y: y['rgb_triplet'])
    acronym_structure_path_map = tree.value_map(lambda x: x['acronym'], lambda y: y['structure_id_path'])
    acronym_structure_id_map = tree.value_map(lambda x: x['acronym'], lambda y: y['id'])
    acronym_structure_graph_order_map = tree.value_map(lambda x: x['acronym'], lambda y: y['graph_order'])
    id_acronym_map = tree.value_map(lambda x: x['id'], lambda y: y['acronym'])

    with open(f"{utils_folder}/summary_structures.pkl", "wb") as fp:
        pickle.dump(summary_structures, fp)

except:
    with open(f"{utils_folder}/acronym_color_map.pkl", "rb") as fp:  # Unpickling
        acronym_color_map = pickle.load(fp)
    with open(f"{utils_folder}/acronym_structure_path_map.pkl", "rb") as fp:  # Unpickling
        acronym_structure_path_map = pickle.load(fp)
    with open(f"{utils_folder}/acronym_structure_id_map.pkl", "rb") as fp:  # Unpickling
        acronym_structure_id_map = pickle.load(fp)
    with open(f"{utils_folder}/acronym_structure_graph_order_map.pkl", "rb") as fp:  # Unpickling
        acronym_structure_graph_order_map = pickle.load(fp)
    with open(f"{utils_folder}/id_acronym_map.pkl", "rb") as fp:  # Unpickling
        id_acronym_map = pickle.load(fp)
    with open(f"{utils_folder}/summary_structures.pkl", "rb") as fp:  # Unpickling
        summary_structures = pickle.load(fp)





def acronym_to_graph_order(area):
    # return (index according to summary_structures(major divisions of mouse brain), and acronym of parent structure)

    if area == "grey":
        graph_order = 2000
    else:
        structure_path = acronym_structure_path_map.get(area)
        parent = set(structure_path).intersection(summary_structures['id']).pop()
        graph_order = summary_structures[summary_structures['id'] == parent]['graph_order'].values[0]

    return graph_order


def acronym_to_main_area(area):
    # return (index according to summary_structures(major divisions of mouse brain), and acronym of parent structure)

    if area == "grey":
        acronym = "grey"
    else:

        structure_path = acronym_structure_path_map.get(area)
        parent = set(structure_path).intersection(summary_structures['id']).pop()
        acronym = summary_structures[summary_structures['id'] == parent]['acronym'].values[0]

    return acronym

def acronym_to_main_area_finer(area):
    # return (index according to summary_structures(major divisions of mouse brain), and acronym of parent structure)

    structure_path = acronym_structure_path_map.get(area)

    if area == "grey":
        acronym = "grey"
    elif len(set(structure_path).intersection(summary_structures_finer['id'])) == 0:
        #print(f"{area} already too generic")
        acronym = area

    else:
        parent = set(structure_path).intersection(summary_structures_finer['id']).pop()
        acronym = summary_structures_finer[summary_structures_finer['id'] == parent]['acronym'].values[0]

    return acronym

def pick_acronym(name, pos):
    # pos is zero, 1 if 2-APN
    s = name.split("-", 2)
    if len(s) == 3:
        name = s[1]
    elif len(s) == 2:
        name = s[pos]
    else:
        name = s[0]
    return name


def color_to_labels(axs, which_axes, minor_or_major, pos=1, *args, **kwargs):
    if which_axes == 'y':

        if minor_or_major == 'minor':
            for ytick in axs.get_yticklabels(minor='True'):
                if ytick.get_text() != 'nan' and ytick.get_text() != []:
                    name = ytick.get_text()
                    name = pick_acronym(name, pos)
                    color = rgb2hex([x / 255 for x in acronym_color_map.get(name)])
                else:
                    color = [0, 0, 0]
                ytick.set_color(color)
        else:
            for ytick in axs.get_yticklabels():
                if ytick.get_text() != 'nan' and ytick.get_text() != []:
                    name = ytick.get_text()
                    name = pick_acronym(name, pos)
                    color = rgb2hex([x / 255 for x in acronym_color_map.get(name)])
                else:
                    color = [0, 0, 0]
                ytick.set_color(color)
    else:
        if minor_or_major == 'minor':
            for xtick in axs.get_xticklabels(minor='True'):
                if xtick.get_text() != 'nan' and xtick.get_text() != []:
                    name = xtick.get_text()
                    name = pick_acronym(name,pos)
                    color = rgb2hex([x / 255 for x in acronym_color_map.get(name)])
                else:
                    color = [0, 0, 0]
                xtick.set_color(color)
        else:
            for xtick in axs.get_xticklabels():
                if xtick.get_text() != 'nan' and xtick.get_text() != []:
                    name = xtick.get_text()
                    name = pick_acronym(name,pos)
                    color = rgb2hex([x / 255 for x in acronym_color_map.get(name)])
                else:
                    color = [0, 0, 0]
                xtick.set_color(color)

def plot_dist_ripple_mod(data, param, ax0):
    g = sns.kdeplot(data=data, x=param,
                    hue='Ripple seed', palette=palette_ML, ax=ax0, fill=True, gridsize=500, cut=0)
    ax0.set_xlim((0, 15))
    ax0.axvline(1,color= 'k', linestyle='--')
    ax0.axvline(2,color= 'r', linestyle='--')
    ax0.get_yaxis().set_visible(False)

    ax0.spines[['left']].set_visible(False)


    norm_test = pg.normality(data=data, dv=param, group="Ripple seed")


    if norm_test["normal"].all():
        p_val = pg.ttest(data[data["Ripple seed"] == "Medial"][param],
               data[data["Ripple seed"] == "Lateral"][param])["p-val"][0]
        print("ttest: ", p_val)

    else:
        p_val = pg.mwu(data[data["Ripple seed"]=="Medial"][param], data[data["Ripple seed"]=="Lateral"][param])["p-val"][0]
        cles = pg.mwu(data[data["Ripple seed"]=="Medial"][param], data[data["Ripple seed"]=="Lateral"][param])["CLES"][0]
        print("mwu p-val and CLES: ", p_val, cles)

    if p_val<.05:
        ax0.text(.6, .7, "*",
                    transform=ax0.transAxes,
                    fontsize=15, ha='center', va='center');
        ax0.text(.6, .8,  f"p-value = {'{:.2e}'.format(p_val)}",
                        transform=ax0.transAxes,
                        fontsize=10, ha='center', va='center');

def point_plot_modulation_ripples(data, dv, parent_area, order, filter_spiking, axs, ylabel, ylim = [.5,3], palette=palette_HPF):

    _ = data[(filter_spiking) & (data['Parent brain region']==parent_area )].reset_index()[['Firing rate','unit_id','M-L',
                                                                                                                    'Session id', 'Brain region', dv + ' medial', dv + ' lateral']]
    _ = pd.wide_to_long(_.reset_index(), stubnames=dv, i=['Brain region','index'], j='Type', sep=' ', suffix=r'\w+').reset_index()
    _['Type'] = _['Type'].str.capitalize()


    ax = sns.pointplot(data=_, x='Brain region', y=dv, hue='Type', dodge=.5, errorbar='se',join=False,  palette=palette_ML, ax=axs, capsize=.2, order=order)
    ax.axhline(1,color= 'k', linestyle='--')
    color_to_labels_custom_palette(ax, 'x', 'major', palette, 1)
    plt.xticks(rotation=45, ha='center')
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)

    stat = []
    for area in _['Brain region'].unique():
        sub = _[_['Brain region']==area]
        if pg.normality(sub, group='Type', dv=dv)['normal'].all():
            test = pg.ttest(sub[sub['Type']=='Medial'][dv], sub[sub['Type']=='Lateral'][dv])['p-val']
        else:
            test = pg.mwu(sub[sub['Type']=='Medial'][dv], sub[sub['Type']=='Lateral'][dv])['p-val']

        if test[0] < .05:
            stat.append(((area, 'Medial'), (area, 'Lateral')))

    if len(stat) > 0:
        annot = Annotator(ax, data=_, pairs=stat,
                          x='Brain region', y=dv, hue='Type', order=order)
        (annot
         .configure(test=None, test_short_name='custom test', text_format='star', loc='inside', verbose=0)
         .set_pvalues(pvalues=[.3] * len(stat))
         .set_custom_annotations(['*'] * len(stat))
         .annotate());
#axs[1].text(.6, .7, 'Cohen's d: ' + str(round(ttest_late_spiking['cohen-d'].values[0], 2)), transform=axs[1].transAxes,fontsize=6, ha='center', va='center');


def color_to_labels_custom_palette(axs, which_axes, minor_or_major, palette, pos=1):
    if which_axes == 'y':

        if minor_or_major == 'minor':
            for ytick in axs.get_yticklabels(minor='True'):
                if ytick.get_text() != 'nan' and ytick.get_text() != []:
                    name = ytick.get_text()
                    print(name)
                    name = pick_acronym(name, pos)
                    color = palette[name]
                else:
                    color = [0, 0, 0]
                ytick.set_color(color)
        else:
            for ytick in axs.get_yticklabels():
                if ytick.get_text() != 'nan' and ytick.get_text() != []:
                    name = ytick.get_text()
                    print(name + "2")
                    name = pick_acronym(name, pos)
                    color = rgb2hex([x / 255 for x in acronym_color_map.get(name)])
                else:
                    color = [0, 0, 0]
                ytick.set_color(color)
    else:
        if minor_or_major == 'minor':
            for xtick in axs.get_xticklabels(minor='True'):
                if xtick.get_text() != 'nan' and xtick.get_text() != []:
                    name = xtick.get_text()
                    name = pick_acronym(name, pos)
                    color = palette[name]
                else:
                    color = [0, 0, 0]
                xtick.set_color(color)
        else:
            for xtick in axs.get_xticklabels():
                if xtick.get_text() != 'nan' and xtick.get_text() != []:
                    name = xtick.get_text()
                    name = pick_acronym(name,pos)
                    color = palette[name]
                else:
                    color = [0, 0, 0]
                xtick.set_color(color)




def plot_animated_ellipse_fits(pupil_data: pd.DataFrame, start_frame: int, end_frame: int):
    start_frame = 0 if (start_frame < 0) else start_frame
    end_frame = len(pupil_data) if (end_frame > len(pupil_data)) else end_frame

    frame_times = pupil_data.index.values[start_frame:end_frame]
    interval = np.average(np.diff(frame_times)) * 1000

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 480), ylim=(0, 480))

    cr_ellipse = Ellipse((0, 0), width=0.0, height=0.0, angle=0, color='white')
    pupil_ellipse = Ellipse((0, 0), width=0.0, height=0.0, angle=0, color='black')
    eye_ellipse = Ellipse((0, 0), width=0.0, height=0.0, angle=0, color='grey')

    ax.add_patch(eye_ellipse)
    ax.add_patch(pupil_ellipse)
    ax.add_patch(cr_ellipse)

    def update_ellipse(ellipse_patch, ellipse_frame_vals: pd.DataFrame, prefix: str):
        ellipse_patch.center = tuple(ellipse_frame_vals[[f"{prefix}_center_x", f"{prefix}_center_y"]].values)
        ellipse_patch.width = ellipse_frame_vals[f"{prefix}_width"]
        ellipse_patch.height = ellipse_frame_vals[f"{prefix}_height"]
        ellipse_patch.angle = np.degrees(ellipse_frame_vals[f"{prefix}_phi"])

    def init():
        return [cr_ellipse, pupil_ellipse, eye_ellipse]

    def animate(i):
        ellipse_frame_vals = pupil_data.iloc[i]

        update_ellipse(cr_ellipse, ellipse_frame_vals, prefix="corneal_reflection")
        update_ellipse(pupil_ellipse, ellipse_frame_vals, prefix="pupil")
        update_ellipse(eye_ellipse, ellipse_frame_vals, prefix="eye")

        return [cr_ellipse, pupil_ellipse, eye_ellipse]

    return animation.FuncAnimation(fig, animate, init_func=init, interval=interval,
                                   frames=range(start_frame, end_frame), blit=True)


def spikes_times_to_bins(area, units, binner, spikes, cutoff_waveform_duration):
    selected_units = units[units['ecephys_structure_acronym'] == area].index
    output = []
    for unit in selected_units:
        X = binner.fit_transform(spikes[spikes['neuron'] == unit])
        if units[units.index == unit]['waveform_duration'].values > cutoff_waveform_duration:
            type = 'exc'
        else:
            type = 'int'

        index = pd.MultiIndex.from_product([[area], [unit], [type]], names=['area', 'id', 'type'])
        X.columns = index
        output.append(X)
    return output


def check_exc_int_per_area_summary(session_number, cache, sessions, cutoff_waveform_duration):
    print('session', session_number, 'start processing')
    session = cache.get_session_data(sessions.index.values[
                                         session_number])  # amplitude_cutoff_maximum = np.inf, presence_ratio_minimum = -np.inf, isi_violations_maximum = np.inf
    units = session.units
    session_numbers = []
    areas = []
    type = []
    number = []
    for area in units['ecephys_structure_acronym'].unique():
        sel = units.loc[units['ecephys_structure_acronym'] == area]
        session_numbers.extend([session_number] * sel.shape[0])
        areas.extend([area] * sel.shape[0])
        number.append(np.count_nonzero(sel['waveform_duration'] < cutoff_waveform_duration))
        type.append('int')
        number.append(np.count_nonzero(sel['waveform_duration'] >= cutoff_waveform_duration))
        type.append('exc')

    summary = pd.DataFrame.from_records([session_numbers, areas, type, number]).T
    summary.columns = ['session', 'area', 'type', 'sum']

    print('session', session_number, 'done!')
    return summary



def bar_plot_percentages(data, ax, title, area_column):
    palette = {area: rgb2hex(np.array(acronym_color_map.get(area)) / 255) for area in area_column.unique()}
    order = ['grey', 'Isocortex', 'CTXsp', 'HPF', 'TH', 'MB', 'HY', 'OLF', 'STR']
    idx = [e in area_column.unique() for e in order]
    order = list(compress(order, idx))
    sns.barplot(data=data, x="main_area", y="session_n", estimator=count_nonzero, alpha=.2, palette=palette,
                order=order, ax=ax)
    g = sns.barplot(data=data, x="main_area", y="correct_prediction", estimator=sum, palette=palette, order=order,
                    ax=ax)
    g.set(title=title)
    for q, p in enumerate(g.patches[len(g.patches) // 2:]):
        area = g.get_xaxis().get_majorticklabels()[q].get_text()
        total = len(data[data["main_area"] == area])
        percentage = f'{100 * p.get_height() / total:.1f}%\n'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        g.annotate(percentage, (x, y+30), ha='center', va='center')


def DL(output, to_use_as_labels, state, how_many):
    # state quiet or run
    collect_res = []
    output_sel = output

    labels, classes_dict = labelize(to_use_as_labels)

    output_sel["labels"] = labels

    for q in tqdm(how_many):
        run_seed = int(q)
        print("seed number", run_seed)
        random.seed(a=run_seed)
        how_many = output["run"].shape[1] / 100 * 75
        sel = random.sample(list(range(len(labels))), k=int(how_many))

        output_norm = output_sel[state] / output_sel[state].max().values
        # for tf 2.
        # train_array = np.expand_dims(output_norm.values[:, sel].T, axis=2)

        train_array = output_norm.values[:, sel].T

        train_labels = output_sel["labels"][sel].values

        test_array = output_norm.values[:, list(set(range(len(labels))) - set(sel))].T

        test_labels = output_sel["labels"][list(set(range(len(labels))) - set(sel))].values

        activation = "selu"

        model = keras.Sequential([

            keras.layers.Dense(128, activation=activation),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(64, activation=activation),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(32, activation=activation),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(len(np.unique(train_labels)), activation='softmax', name='output')
        ])

        model.compile(optimizer=tf.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        weights = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.unique(train_labels),
                                                                  y=train_labels)
        weights = {q: weights[p] for p, q in enumerate(np.unique(train_labels))}

        #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

        history = model.fit(train_array, train_labels, batch_size=64, epochs=3200, class_weight=weights,
                            validation_data=(test_array, test_labels),  verbose=0)

        #plt.plot(history.history['loss'])
        #plt.plot(history.history['val_loss'])
       # plt.plot(history.history['acc'])
        #plt.plot(history.history['val_acc'])

        test_loss, test_acc = model.evaluate(test_array, test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)

        predictions = np.argmax(model.predict(test_array),1)

        test = output_sel["run"][:, list(set(range(len(labels))) - set(sel))]
        test = test.assign_coords(prediction=("area", predictions))
        test = test.assign_coords(prediction_error=("area", np.invert(test_labels == predictions)))
        test = test.assign_coords(
            correct_prediction=("area", test_labels == predictions))
        test = test.assign_coords(main_area=("area", [acronym_to_main_structure(x) for x in test["area"].values]))

        res = test.coords["area"].to_dataframe()
        res.reset_index(inplace=True, drop=True)
        res["run_seed"] = run_seed
        collect_res.append(res)
    return collect_res

def labelize(to_label):
    classes = np.unique(to_label)
    classes_dict = {v: k for k, v in enumerate(classes)}
    labels = [classes_dict.get(key) for key in to_label]
    return labels, classes_dict

def closestNumber(n, m):
    # Find the quotient
    q = int(n / m)

    # 1st possible closest number
    n1 = m * q

    # 2nd possible closest number
    if (n * m) > 0:
        n2 = (m * (q + 1))
    else:
        n2 = (m * (q - 1))

    # else n2 is the required closest number
    return n1, n2


def ripple_finder(sig, fs, threshold_ripples, probe_n, brain_area, space_sub_spike_times):

    filtered = butter_bandpass_filter(np.nan_to_num(sig.values), lowcut, highcut, fs, order=6)
    analytic_signal = hilbert(filtered)
    amplitude_envelope = np.abs(analytic_signal)
    amplitude_envelope = pd.Series(amplitude_envelope, index=sig.time)

    if acronym_to_graph_order(brain_area) == 454:  # if area belongs to HPF
        print("HPF specific threshold")
        threshold = np.std(amplitude_envelope) * ripple_thr
    else:
        print("generic threshold")
        threshold = threshold_ripples

    unwrapped = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(medfilt(unwrapped, int(fs / 1000 * 12)))  # according to tingsley buzsaky 2021
    instantaneous_frequency = instantaneous_frequency / (2 * np.pi) * fs

    peaks, peak_properties = signal.find_peaks(amplitude_envelope, height=threshold, distance=fs/1000*50)

    peak_heights = peak_properties["peak_heights"]

    prominences, left_bases, right_bases = signal.peak_prominences(amplitude_envelope, peaks)
    results_half = signal.peak_widths(amplitude_envelope.rolling(5, center=True).mean(), peaks, rel_height=1, prominence_data=(
        peak_heights - np.std(amplitude_envelope) * thr_start_stop, left_bases, right_bases))  # width evaluated at: np.std(amplitude_envelope) * 2
    peaks_width_sec = results_half[0] / fs
    mask = (peaks_width_sec > ripple_dur_lim[0]) & (peaks_width_sec < ripple_dur_lim[1])  # (peaks_width_sec > 0.015)   & (peaks_width_sec < 0.250) Tingley
    peak_heights = peak_heights[mask]

    peaks_sec = sig.time.values[peaks][mask]
    peaks_start_sec = sig.time.values[np.rint(results_half[2]).astype(int)][mask]
    peaks_stop_sec = sig.time.values[np.rint(results_half[3]).astype(int)][mask]

    array_diff = np.diff(peaks_start_sec) > min_ripple_distance # if nearer than x keep first one
    if array_diff.size != 0:
        array_diff[-1] = True
        start_mask = clean_start_time_ripples(array_diff)
        peak_mask = clean_peak_time_ripples(array_diff, peak_heights)
        peaks_start_sec = peaks_start_sec[start_mask]
        stop_mask = list(array_diff)
        stop_mask.append(True)
        peaks_stop_sec = peaks_stop_sec[stop_mask]
        temp_peak1 = peaks_sec[stop_mask]
        temp_peak2 = peaks_sec[peak_mask]
        peaks_sec = np.maximum(temp_peak1, temp_peak2)
        # temp_height1 = peak_heights[stop_mask]
        # temp_height2 = peak_heights[peak_mask]
        # peak_heights = np.maximum(temp_height1, temp_height2)

        ripples = pd.DataFrame([peaks_sec, peaks_start_sec, peaks_stop_sec],
                               index=["Peak (s)", "Start (s)", "Stop (s)"]).T

        ripples["Probe number"] = probe_n
        ripples["Area"] = brain_area
        ripples["Duration (s)"] = ripples["Stop (s)"] - ripples["Start (s)"]

        instantaneous_frequency_pd = pd.DataFrame(instantaneous_frequency, index=sig.time[:-1].values, columns=["freq"])

        inst_freq = []
        for index, row in ripples.iterrows():
            try:
                inst_freq.append(np.mean(instantaneous_frequency_pd[row["Start (s)"]: row["Stop (s)"]].values))
            except:
                inst_freq.append(np.nan)

        ripples["Instantaneous Frequency (Hz)"] = inst_freq
        ripples["Probe number-area"] = ripples[["Probe number", "Area"]].apply(lambda row: '-'.join(row.values.astype(str)), axis=1)

        amplitude_envelope = pd.Series(amplitude_envelope, index=sig.time)
        #filtered = pd.Series( filtered, index=sig.time)
        sig = sig.to_series().fillna(value=0)

        high_freq_integral = []
        power_peak = []
        freq_peak = []
        amplitude = []
        number_spikes_list = []
        number_participating_neurons_list = []

        ripples_start = ripples["Start (s)"]
        ripples_stop = ripples["Stop (s)"]

        for start, stop in zip(ripples_start, ripples_stop):
            time_space_sub_spike_times = {
                cluster_id: spikes[(spikes > start) & (spikes < stop)] for
                cluster_id, spikes in space_sub_spike_times.items()}

            number_spikes, number_participating_neurons = process_spike_and_neuron_numbers_per_ripple(
                time_space_sub_spike_times)
            number_spikes_list.append(number_spikes)
            number_participating_neurons_list.append(number_participating_neurons)

            ripple_lfp = sig[(sig.index > start) & (sig.index < stop)]
            f, pxx = signal.periodogram(ripple_lfp, fs=fs, detrend="constant", scaling="spectrum")
            peaks, properties = signal.find_peaks(pxx, prominence=[0.002, None])
            if len(peaks) > 1:
                peaks = peaks.max()
            elif len(peaks) == 1:
                peaks = peaks[0]

            freq_peak.append(f[peaks])
            power_peak.append(pxx[peaks]*1000)
            high_freq_integral.append(np.trapz(amplitude_envelope[(amplitude_envelope.index > start) & (amplitude_envelope.index < stop)]))
            amplitude.append(amplitude_envelope[(amplitude_envelope.index > start) & (amplitude_envelope.index < stop)].quantile(.9))

        freq_peak = [el if isinstance(el, (int, float)) else np.nan for el in
                     freq_peak]  # don't want empty lists in the series
        power_peak = [el if isinstance(el, (int, float)) else np.nan for el in power_peak]

        ripples["∫Ripple"] = high_freq_integral
        ripples["Peak frequency (Hz)"] = freq_peak
        ripples["Peak power"] = power_peak
        ripples["Amplitude (mV)"] = amplitude
        ripples["Number spikes"] = number_spikes_list
        ripples["Number participating neurons"] = number_participating_neurons_list

        ripples = ripples[~ripples["Start (s)"].duplicated(keep="last")]

        # filter by peak freq
        print(probe_n, brain_area, "Ripples retained by peak freq: ", ripples[ripples["Peak frequency (Hz)"] > 100].shape[0],
              ", total: ", ripples.shape[0])
        ripples = ripples[ripples["Peak frequency (Hz)"] > ripple_power_sp_thr]

        print("Duplicated starts: " + str(
            np.sum(ripples["Start (s)"].duplicated(keep="last"))) + ", Duplicated stops: " + str(
            np.sum(ripples["Stop (s)"].duplicated())))

        print(probe_n, brain_area, "Ripples discarded: ", sum(~mask))# not counting joined ones

        print(probe_n, brain_area, "Ripples detected: ", len(peaks_sec))
    else:
        ripples = pd.DataFrame()
        print(probe_n, brain_area, "no ripples detected!")

    return ripples


def clean_start_time_ripples(array_diff):
    # if closer than x keep first starting time
    start_mask = []
    flag_next = False
    for key, group in groupby(array_diff):
        # print(key, list(group))
        g = list(group)
        if key == False:
            g[0] = True
            flag_next = True
            start_mask.extend(g)
        elif flag_next == True:
            g[0] = False
            start_mask.extend(g)
            flag_next = False
        else:
            start_mask.extend(g)

    start_mask.append(True)

    return start_mask


def clean_peak_time_ripples(array_diff, peak_heights):
    #pick highest peak if ripples closer than x
    flag_next = False
    peak_mask = []
    for key, group in groupby(zip(array_diff, peak_heights), itemgetter(0)):
        g = list(list(zip(*group))[1])
        gg = ([key] * len(g))
        if key == False:
            gg[np.argmax(g)] = True
            flag_next = True
            peak_mask.extend(gg)
        elif flag_next == True:
            gg[0] = False
            peak_mask.extend(gg)
            flag_next = False
        else:
            peak_mask.extend(gg)

    peak_mask.append(True)
    return peak_mask


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output="sos")
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y

def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def calculations_per_ripple(start, stop, sliced_lfp_per_probe, lowcut, highcut, fs_lfp, length):

    high_freq_area = []
    areas = []
    probe_n_area = []
    AUC_pos = []
    AUC_neg = []

    for probe_n, lfp in enumerate(sliced_lfp_per_probe):

        for sig in lfp.T:
            if sig.shape[0] == length:
                sig.values = np.nan_to_num(sig.values)
                filtered = butter_bandpass_filter(sig.values, lowcut, highcut, fs_lfp, order=6)
                analytic_signal = hilbert(filtered)
                amplitude_envelope = np.abs(analytic_signal)
                amplitude_envelope = pd.Series(amplitude_envelope, index=sig.time)
                high_freq_area.append(np.trapz(amplitude_envelope
                                               [(amplitude_envelope.index > start - 0.02) & (amplitude_envelope.index < stop + 0.02)]))
                # for AUC
                sig = sig.sel(time=slice(start, start + 0.2))
                AUC_pos.append(np.trapz(sig.values[sig.values > 0]))
                AUC_neg.append(np.trapz(sig.values[sig.values < 0]))
            else:
                #print(f"LFP from {sig.area.values} at {timestamp} is {sig.shape[0]} instead of {length} samples long.")
                high_freq_area.append(np.nan)
                AUC_pos.append(np.nan)
                AUC_neg.append(np.nan)

        areas.append(lfp.area.values)

        for area in lfp.area.values:
            probe_n_area.append((probe_n, area))

    res = pd.DataFrame([high_freq_area, AUC_pos, AUC_neg],
                       columns=pd.MultiIndex.from_tuples(probe_n_area, names=["probe_n", "area"]),
                       index=["Ripple area (mV*s)", "Positive area (mV*s)", "Negative area (mV*s)"])

    sorting_df = pd.DataFrame(
        [np.concatenate(areas), [acronym_to_graph_order(area.split("-")[0]) for area in np.concatenate(areas)]],
        index=["area", "graph_id"]).T
    idx = sorting_df.sort_values(["graph_id", "area"]).index
    res = res.iloc[:, idx]

    return res, start

def std_skew_on_filtered(trace, fs_lfp):
    lowcut = 120.0
    highcut = 250.0
    filtered = butter_bandpass_filter(np.nan_to_num(trace), lowcut, highcut, fs_lfp, order=6)
    analytic_signal = hilbert(filtered)
    amplitude_envelope = np.abs(analytic_signal).T

    return np.stack((np.std(amplitude_envelope), skew(amplitude_envelope)))

def select_quiet_part(lfp, start_quiet, stop_quiet):
    out = []
    for start, stop in zip(start_quiet, stop_quiet):
        out.append(lfp.sel(time=slice(start, stop)))
    out = xr.concat(out, dim="time")
    return out



def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f"r={round(r, 3)}", xy=(.01, .9), xycoords=ax.transAxes)


def R2func(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f"R\u00b2={round(r ** 2, 3)}", xy=(.01, .9), xycoords=ax.transAxes)

def is_outlier(s):
    lower_limit = s.mean() - (s.std() * 50)
    upper_limit = s.mean() + (s.std() * 5)
    return ~s.between(lower_limit, upper_limit)

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def calc_sync(n, reference_probe, starts):
    out = []
    out2 = []

    for s in starts[reference_probe].dropna().iteritems():
        nearest = find_nearest(starts[n].dropna().values, s[1])

        out.append(np.min(np.abs(starts[n].dropna() - s[1])) < 0.05)
        out2.append(nearest - s[1])

    return out, out2

def clean_ripples_calculations(ripples_calcs):
    """
    clean from mistakes: one case with negative dv,ap and lr coords, "grey" (unassigned) areas and ripples during
    run.
    """

    for session_id, session_data in tqdm(ripples_calcs.items()):
        print(f"Session {session_id}")

        pos = session_data[1]
        sel_probe = session_data[5]

        out = []
        for t in session_data[0][0].columns:
            out.append(pos[(pos["Probe number"] == t[0]) & (pos["Area"] == t[1])])

        pos = pd.concat(out).reset_index(drop=True)

        mask = (pos["D-V (µm)"].values > 0) & (session_data[0][0].columns.get_level_values(1) != "grey")
        ripples_calcs[session_id][0][0] = session_data[0][0].loc[:, session_data[0][0].columns[mask]]
        ripples_calcs[session_id][0][1] = session_data[0][1].loc[:, session_data[0][1].columns[mask]]
        ripples_calcs[session_id][0][2] = session_data[0][2].loc[:, session_data[0][2].columns[mask]]
        # create M-L axis, subtract midpoint
        pos["M-L (µm)"] = pos["L-R (µm)"] - 5691.510009765625
        ripples_calcs[session_id][1] = pos.iloc[pos.index[mask]]

       # clean from running epochs

        start_running = ripples_calcs[session_id][4][1]
        stop_running = ripples_calcs[session_id][4][2]
        behavior = ripples_calcs[session_id][4][0]

        ripple_power = ripples_calcs[session_id][0][0]
        pos_area = ripples_calcs[session_id][0][1]
        neg_area = ripples_calcs[session_id][0][2]

        #  clean ripples if behavior data not available
        for q in range(len(start_running)):
            idxs = ripple_power.loc[
                   ripple_power.index[(ripple_power.index > start_running[q]) & (ripple_power.index < stop_running[q])],
                   :].index
            if idxs.shape[0] > 0:
                pos_area.drop(idxs, inplace=True)
                neg_area.drop(idxs, inplace=True)
                ripple_power.drop(idxs, inplace=True)

        idxs = ripple_power.loc[
               ripple_power.index[(ripple_power.index > behavior.iloc[-1].name) | (ripple_power.index < behavior.iloc[0].name)],
               :].index

        if idxs.shape[0] > 0:
            pos_area.drop(idxs, inplace=True)
            neg_area.drop(idxs, inplace=True)
            ripple_power.drop(idxs, inplace=True)

        ripples = ripples_calcs[session_id][3]

        #ripples = ripples[ripples["Peak frequency (Hz)"]> 120]

        # create M-L axis, subtract midpoint
        ripples["M-L (µm)"] = ripples["L-R (µm)"].copy() - 5691.510009765625

        ripples_calcs[session_id][3] = clean_ripples_from_running(ripples, start_running, stop_running, behavior)
        print(f"{session_id}: ripples number on best probe:{ripples[ripples['Probe number']==sel_probe].shape[0]}")

    return ripples_calcs


def clean_ripples_from_running(ripples, start_running, stop_running, behavior):
    for q in range(len(start_running)):
        idxs = ripples.loc[
               ripples.index[(ripples["Start (s)"] > start_running[q]) & (ripples["Start (s)"] < stop_running[q])],
               :].index
        if idxs.shape[0] > 0:
            ripples.drop(idxs, inplace=True)

    #  clean ripples if behavior data not available
    idxs = ripples.loc[
           ripples.index[(ripples["Start (s)"] > behavior.iloc[-1].name) | (ripples["Start (s)"] < behavior.iloc[0].name)],
           :].index
    if idxs.shape[0] > 0:
        ripples.drop(idxs, inplace=True)

    return ripples.reset_index(drop=True)


def find_couples_based_on_distance(summary_corrs, ripples_calcs, sessions, var_thr):

    print(f"Threshold var = {var_thr}")

    quartiles_distance = summary_corrs[summary_corrs["Comparison"] == "CA1-CA1"]["Distance (µm)"].quantile(
        [0.25, 0.5, 0.75])

    high_distance = []
    low_distance = []
    distance_tabs = []

    for session_id, sel in ripples_calcs.items():

        ripples = sel[3].copy()

        ripples = ripples[ripples["Area"] == "CA1"]

        ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)

        if "CA1" in ripples["Area"].unique():

            pos_matrix = ripples.groupby("Probe number-area").mean()[["D-V (µm)", "A-P (µm)", "M-L (µm)"]].T

            distance_matrix = pos_matrix.corr(method=calculate_distance)
            distance_tab = distance_matrix.where(np.triu(np.ones(distance_matrix.shape)).astype(np.bool))
            distance_tab.columns.name = None
            distance_tab.index.name = None

            distance_matrix_dv = pos_matrix.corr(method=calculate_distance_dv)
            distance_tab_dv = distance_matrix_dv.where(np.triu(np.ones(distance_matrix_dv.shape)).astype(np.bool))
            distance_tab_dv.columns.name = None
            distance_tab_dv.index.name = None

            distance_matrix_ap = pos_matrix.corr(method=calculate_distance_ap)
            distance_tab_ap = distance_matrix_ap.where(np.triu(np.ones(distance_matrix_ap.shape)).astype(np.bool))
            distance_tab_ap.columns.name = None
            distance_tab_ap.index.name = None

            distance_matrix_ml = pos_matrix.corr(method=calculate_distance_ml)
            distance_tab_ml = distance_matrix_ml.where(np.triu(np.ones(distance_matrix_ml.shape)).astype(np.bool))
            distance_tab_ml.columns.name = None
            distance_tab_ml.index.name = None

            distance_tab = distance_tab.stack().reset_index()
            distance_tab_dv = distance_tab_dv.stack().reset_index()
            distance_tab_ap = distance_tab_ap.stack().reset_index()
            distance_tab_ml = distance_tab_ml.stack().reset_index()

            distance_tab.columns = ['Reference area', 'Secondary area', 'Distance (µm)']
            distance_tab = pd.concat([distance_tab,  distance_tab_dv.iloc[:,2].rename("D-V (µm)"),
                                       distance_tab_ap.iloc[:,2].rename("A-P (µm)"),
                                       distance_tab_ml.iloc[:,2].rename("M-L (µm)")], axis=1)

            distance_tab = distance_tab[distance_tab["Distance (µm)"] > 1]

            high_distance_tab = distance_tab[distance_tab["Distance (µm)"] > quartiles_distance[0.75]]
            low_distance_tab = distance_tab[distance_tab["Distance (µm)"] < quartiles_distance[0.25]]

            if high_distance_tab.shape[0] > 0 and low_distance_tab.shape[0] > 0 and \
                    ripples[ripples["Probe number"] == sel[5]].shape[0] > minimum_ripples_count_lag_analysis:
                high_distance_tab = high_distance_tab.loc[high_distance_tab['Distance (µm)'].idxmax(), :]
                selected_ripples = ripples[(ripples["Probe number-area"] == high_distance_tab["Reference area"]) |
                                           (ripples["Probe number-area"] == high_distance_tab[
                                               "Secondary area"])].reset_index(drop=True)

                selected_ripples["Distance (µm)"] = high_distance_tab['Distance (µm)']
                selected_ripples["Session"] = session_id
                selected_ripples["Sex"] = sessions.loc[session_id]["sex"]
                selected_ripples["Age"] = sessions.loc[session_id]["age_in_days"]

                high_distance.append(selected_ripples)

                low_distance_tab = low_distance_tab.loc[low_distance_tab['Distance (µm)'].idxmin(), :]
                selected_ripples = ripples[(ripples["Probe number-area"] == low_distance_tab["Reference area"]) |
                                           (ripples["Probe number-area"] == low_distance_tab[
                                               "Secondary area"])].reset_index(drop=True)

                selected_ripples["Distance (µm)"] = low_distance_tab['Distance (µm)']

                selected_ripples["Session"] = session_id
                selected_ripples["Sex"] = sessions.loc[session_id]["sex"]
                selected_ripples["Age"] = sessions.loc[session_id]["age_in_days"]

                low_distance.append(selected_ripples)
                distance_tabs.append(distance_tab)

    distance_tabs = pd.concat(distance_tabs)
    high_distance = pd.concat(high_distance).reset_index(drop=True)
    low_distance = pd.concat(low_distance).reset_index(drop=True)

    print("Computed distances")

    return high_distance, low_distance, distance_tabs

def calculate_distance(x, y):
    return np.linalg.norm(x-y)

def calculate_distance_dv(x, y):
    return np.abs(x[0] - y[0] )

def calculate_distance_ap(x, y):
    return np.abs(x[1] - y[1])

def calculate_distance_ml(x, y):
    return np.abs(x[2] - y[2])

def calculate_corrs(ripples_calcs, sessions, var_thr):

    print(f"Threshold var = {var_thr}")

    out = []

    for session_id, sel in tqdm_notebook(ripples_calcs.items()):

        ripple_power = sel[0][0].copy()
        ripple_power = ripple_power.loc[:, ripple_power.var() > var_thr]  # absolute threshold found empirically
        # ripple_power = ripple_power[ripple_power.columns[ripple_power.columns.get_level_values(1)=="CA1"]]

        if ripple_power.shape[1] > 1:

            ripple_power.columns = ['{}-{}'.format(i[0], i[1]) for i in ripple_power.columns]
            a = ripple_power.corr()
            a.columns.name = "Secondary area"
            a.index.name = "Reference area"
            a = a.where(np.triu(np.ones(a.shape)).astype(np.bool))
            np.fill_diagonal(a.values, np.nan)
            aa = a.stack().reset_index()

            aa.columns = ["Reference area", "Secondary area", "Correlation"]
            aa["Session"] = session_id
            aa["Sex"] = sessions.loc[session_id]["sex"]

            pos = sel[1]
            t_out = []
            for t in aa["Reference area"]:
                t_out.append(
                    pos[(pos["Probe number"] == int(t.split("-", 1)[0])) & (pos["Area"] == t.split("-", 1)[1])])
            pos_area1 = pd.concat(t_out).reset_index(drop=True).iloc[:, 2:]

            t_out = []
            for t in aa["Secondary area"]:
                t_out.append(
                    pos[(pos["Probe number"] == int(t.split("-", 1)[0])) & (pos["Area"] == t.split("-", 1)[1])])
            pos_area2 = pd.concat(t_out).reset_index(drop=True)[["D-V (µm)", "A-P (µm)", "L-R (µm)", "M-L (µm)"]]
            pos_area2.columns = ["D-V2 (µm)", "A-P2 (µm)", "L-R2 (µm)", "M-L2 (µm)"]

            _ = pd.concat([aa, pos_area1, pos_area2], axis=1).infer_objects()
            _["Distance (µm)"] = np.sqrt((_["A-P (µm)"] - _["A-P2 (µm)"]) ** 2 + (_["D-V (µm)"] - _["D-V2 (µm)"]) ** 2 + (
                        _["M-L (µm)"] - _["M-L2 (µm)"]) ** 2)
            _["Distance A-P (µm)"] = np.sqrt((_["A-P (µm)"] - _["A-P2 (µm)"]) ** 2)
            _["Distance D-V (µm)"] = np.sqrt((_["D-V (µm)"] - _["D-V2 (µm)"]) ** 2)
            _["Distance M-L (µm)"] = np.sqrt((_["M-L (µm)"] - _["M-L2 (µm)"]) ** 2)

            out.append(_)

    summary_corrs = pd.concat(out, axis=0).reset_index(drop=True)

    summary_corrs["Area 1"] = summary_corrs["Reference area"].str.split("-", 1, expand=True).iloc[:, 1]
    summary_corrs["Area 2"] = summary_corrs["Secondary area"].str.split("-", 1, expand=True).iloc[:, 1]
    summary_corrs["Comparison"] = summary_corrs["Area 1"] + "-" + summary_corrs["Area 2"]
    summary_corrs['Count'] = summary_corrs.groupby('Comparison')['Comparison'].transform('count')

    return summary_corrs

def calculate_lags(high_distance, low_distance, sessions, invert_reference):
    out = []

    for n in tqdm(high_distance["Session"].unique()):

        data = high_distance[high_distance["Session"] == n]
        _ = data.pivot(columns=['Probe number', "M-L (µm)"], values=['Start (s)', '∫Ripple', "Duration (s)"]).reset_index(drop=True)
        if invert_reference == True:
            _ = _.iloc[:, np.argsort(_.columns.get_level_values(1))[[3,4,5,0,1,2]]]  #2, 3, 0, 1 order by LR or ML
        else:
            _ = _.iloc[:, np.argsort(_.columns.get_level_values(1))]  # order by LR or ML
        times_reference = _.iloc[:, 0].dropna().reset_index(drop=True)
        times_secondary = _.iloc[:, 3].dropna().reset_index(drop=True)
        ripple_area_primary = _.iloc[:, 1].dropna().reset_index(drop=True)
        duration_primary = _.iloc[:, 2].dropna().reset_index(drop=True)
        ripple_area_secondary = []

        out_diffs = []

        for val in times_reference:
            nearest = find_nearest(times_secondary, val)
            ripple_area_secondary.append(_[_['Start (s)'].iloc[:, 1] == nearest].iloc[:, 3].values[0])

            out_diffs.append((-val + nearest) * 1000)  # val as zero

        diffs_high = pd.DataFrame(out_diffs, columns=["Lag (ms)"])
        diffs_high["Type"] = "High distance (µm)"
        diffs_high["Distance (µm)"] = data["Distance (µm)"].iloc[0]
        diffs_high['Start (s)'] = times_reference
        diffs_high['∫Ripple'] = ripple_area_primary
        diffs_high['Duration (s)'] = duration_primary
        diffs_high['LR 1'] = _.columns.get_level_values(2)[0]
        diffs_high['LR 2'] = _.columns.get_level_values(2)[2]
        diffs_high['∫Ripple 2'] = ripple_area_secondary

        data = low_distance[low_distance["Session"] == n]
        _ = data.pivot(columns=['Probe number', "M-L (µm)"], values=['Start (s)', '∫Ripple' , "Duration (s)"]).reset_index(drop=True)
        if invert_reference == True:
            _ = _.iloc[:, np.argsort(_.columns.get_level_values(1))[[3,4,5,0,1,2]]]  # order by LR or ML
        else:
            _ = _.iloc[:, np.argsort(_.columns.get_level_values(1))]  # order by LR or ML
        times_reference = _.iloc[:, 0].dropna().reset_index(drop=True)
        times_secondary = _.iloc[:, 3].dropna().reset_index(drop=True)
        ripple_area_primary = _.iloc[:, 1].dropna().reset_index(drop=True)
        duration_primary = _.iloc[:, 2].dropna().reset_index(drop=True)
        ripple_area_secondary = []

        out_diffs = []

        for val in times_reference:
            nearest = find_nearest(times_secondary, val)
            ripple_area_secondary.append(_[_['Start (s)'].iloc[:, 1] == nearest].iloc[:, 3].values[0])
            out_diffs.append((-val + nearest) * 1000)

        diffs_low = pd.DataFrame(out_diffs, columns=["Lag (ms)"])
        diffs_low["Type"] = "Low distance (µm)"
        diffs_low["Distance (µm)"] = data["Distance (µm)"].iloc[0]
        diffs_low['Start (s)'] = times_reference
        diffs_low['∫Ripple'] = ripple_area_primary
        diffs_low['Duration (s)'] = duration_primary
        diffs_low['LR 1'] = _.columns.get_level_values(2)[0]
        diffs_low['LR 2'] = _.columns.get_level_values(2)[2]
        diffs_low['∫Ripple 2'] = ripple_area_secondary

        diffs = pd.concat([diffs_high, diffs_low], axis=0).reset_index(drop=True)
        diffs["Session"] = n
        diffs["Sex"] = sessions.loc[n]["sex"]

        out.append(diffs)

    tot_diffs = pd.concat(out).reset_index(drop=True)
    tot_diffs["Absolute lag (ms)"] = np.abs(tot_diffs["Lag (ms)"])

    ripples_lags = tot_diffs

    print(f"Computed lags, {ripples_lags.Session.unique().shape[0]} sessions retained")

    return ripples_lags

def find_ripples_clusters_new(ripples, source_area):
    """
    Reorganize ripples relative to one particular probe. Cluster together ripples happening in a window as the same ripple
    travelling along the hippocampus.
    """

    out = []

    ripples_source_area = ripples[ripples["Probe number-area"] == source_area]

    for ripple_source in tqdm(ripples_source_area.iterrows(), total=ripples_source_area.shape[0]):

        near_ripples = ripples.loc[np.abs(ripples["Start (s)"] - ripple_source[1]["Start (s)"]) < thr_rip_cluster].copy()
        near_ripples["Start (s)"] = near_ripples["Start (s)"] - ripple_source[1]["Start (s)"]

        # if detected twice on source probe we keep the source (0.00 s)
        if near_ripples[near_ripples["Probe number-area"] == source_area].shape[0] > 1:
            to_drop = near_ripples[
                (near_ripples["Probe number-area"] == source_area) & (near_ripples["Start (s)"] != 0)].index
            near_ripples = near_ripples.drop(to_drop)

        # if detected twice or more on one probe we keep the first one temporally
        if np.any(near_ripples["Probe number-area"].duplicated(keep=False)):
            to_drop = near_ripples.index[near_ripples["Probe number-area"].duplicated(keep=False)]
            to_drop = to_drop.drop(
                near_ripples[near_ripples["Probe number-area"].duplicated(keep=False)]["Start (s)"].idxmin())
            near_ripples = near_ripples.drop(to_drop)

        # print(near_ripples)

        positions = near_ripples[["D-V (µm)", "A-P (µm)", "M-L (µm)"]].copy()
        distances = positions.apply(lambda row: calculate_distance(
            near_ripples[near_ripples["Start (s)"] == 0][["D-V (µm)", "A-P (µm)", "M-L (µm)"]], row[["D-V (µm)", "A-P (µm)", "M-L (µm)"]]), axis=1)

        velocities = distances / (near_ripples["Start (s)"].copy().abs() * 1000)

        strengths = near_ripples["Z-scored ∫Ripple"].copy()

        velocities.index = near_ripples["Probe number-area"].copy() + " speed (µm/ms)"
        distances.index = near_ripples["Probe number-area"].copy() + " distance (µm)"
        strengths.index = near_ripples["Probe number-area"].copy() + " Z-scored ∫Ripple"

        ripple_numbers = near_ripples["Ripple number"].copy()
        ripple_numbers.index = near_ripples["Probe number-area"] + " ripple number"

        _ = pd.Series(near_ripples["Start (s)"]).copy()
        _.name = ripple_source[1]["Ripple number"]

        _.index = near_ripples["Probe number-area"] + " lag (s)"
        _["∫Ripple"] = ripple_source[1]["∫Ripple"]
        _["Duration (s)"] = ripple_source[1]["Duration (s)"]
        _["Start (s)"] = ripple_source[1]["Start (s)"]
        _["Strongest ripple M-L (µm)"] = near_ripples.loc[near_ripples["Z-scored ∫Ripple"].idxmax()]["M-L (µm)"]
        _["Spatial engagement"] = (near_ripples.shape[0]-1) / (ripples["Probe number-area"].unique().shape[0]-1)

        _["Source M-L (µm)"] = near_ripples.loc[near_ripples["Start (s)"].idxmin()]["M-L (µm)"]

        #if near_ripples.shape[0] > 1:
            # # linear regression to determine direction
            # y = near_ripples["M-L (µm)"]
            # x = near_ripples["Start (s)"]
            # fit = np.polyfit(x, y, deg=1)
            # # predict = np.poly1d(fit)
            # # plt.plot(x, predict(x), color="k", alpha=0.3)
            # # plt.scatter(x, y)
            #
            # if fit[0] > 0:
            #     _["Direction"] = "M\u2192L"
            # elif fit[0] < 0:
            #     _["Direction"] = "L\u2192M"

        # strenght across M-L axis
        if near_ripples.loc[near_ripples["Start (s)"].idxmin()]["Probe number-area"] == source_area:
            _["Global strength"] = near_ripples["Local strong"].sum()/ripples["Probe number-area"].unique().shape[0]
        else:
            _["Global strength"] = np.nan

        if near_ripples.loc[near_ripples["Start (s)"].idxmin()]["Probe number-area"] == source_area:
            _["Source"] = True
        else:
            _["Source"] = False
        _2 = pd.concat([_, velocities, strengths, distances, ripple_numbers])

        out.append(_2)
    _ = pd.concat(out, axis=1).T
    _.index.name = "Ripple number"
    _.columns.name = ""

    return _.reindex(sorted(_.columns), axis=1)


def find_ripples_clusters_new_total(ripples, session_id):
    thr_rip = 0.06  # belong to same ripple if closer than
    thr_sep = 0.06  # minimum separation between clusters
    out = []

    idx = 0
    pbar = tqdm_notebook(total=ripples.shape[0] - 1, initial=idx)

    # TODO: add strong column, norm for each probe
    while 1:

        ripple_source = ripples.loc[idx]
        source_area = ripple_source["Probe number-area"]

        near_ripples = ripples.loc[np.abs(ripples["Start (s)"] - ripple_source["Start (s)"]) < thr_rip].copy()
        near_ripples["Start (s)"] = near_ripples["Start (s)"] - ripple_source["Start (s)"]

        # if detected twice on source probe we keep the source (0.00 s)
        if near_ripples[near_ripples["Probe number-area"] == source_area].shape[0] > 1:
            to_drop = near_ripples[
                (near_ripples["Probe number-area"] == source_area) & (near_ripples["Start (s)"] != 0)].index
            near_ripples = near_ripples.drop(to_drop)

        # if detected twice or more on one probe we keep the first one temporally
        if np.any(near_ripples["Probe number-area"].duplicated(keep=False)):
            to_drop = near_ripples.index[near_ripples["Probe number-area"].duplicated(keep=False)]
            to_drop = to_drop.drop(
                near_ripples[near_ripples["Probe number-area"].duplicated(keep=False)]["Start (s)"].idxmin())
            near_ripples = near_ripples.drop(to_drop)

        # print(near_ripples)

        positions = near_ripples[["D-V (µm)", "A-P (µm)", "L-R (µm)"]]
        # distances = positions.apply(lambda row: calculate_distance(
        #     near_ripples[near_ripples["Start (s)"] == 0][["DV (µm)", "AP (µm)", "LR (µm)"]], row[["DV (µm)", "AP (µm)", "LR (µm)"]]), axis=1)
        #
        #
        # velocities = distances / (near_ripples["Start (s)"].abs() * 1000)
        # velocities.index = near_ripples["Probe number-area"] + " speed (µm/ms)"
        # distances.index = near_ripples["Probe number-area"] + " distance (µm)"

        _ = pd.Series(near_ripples["Start (s)"]).copy()

        _.index = near_ripples["Probe number-area"] + " lag (s)"
        _["∫Ripple"] = ripple_source["∫Ripple"]
        _["Start (s)"] = ripple_source["Start (s)"]
        _["Strongest ripple LR (µm)"] = near_ripples.loc[near_ripples["Z-scored ∫Ripple"].idxmax()]["LR (µm)"]
        _["Spatial engagement"] = near_ripples.shape[0] / ripples["Probe number-area"].unique().shape[0]

        _["Source LR (µm)"] = near_ripples.loc[near_ripples["Start (s)"].idxmin()]["LR (µm)"]
        _["Source Ripple number"] = idx


        # linear regression to determine direction
        if near_ripples.shape[0] > 1:
            y = near_ripples["LR (µm)"]
            x = near_ripples["Start (s)"]
            fit = np.polyfit(x, y, deg=1)
            # predict = np.poly1d(fit)
            # plt.plot(x, predict(x), color="k", alpha=0.3)
            # plt.scatter(x, y)

            if fit[0] > 0:
                _["Direction"] = "M\u2192L"
            elif fit[0] < 0:
                _["Direction"] = "L\u2192M"
        else:
            _["Direction"] = np.nan

        # _ = pd.concat([_, velocities, distances])

        idx = near_ripples.index[-1] + 1

        if idx >= ripples.shape[0]:
            break

        while (np.abs(ripples.loc[idx]["Start (s)"] - ripples.loc[idx - 1]["Start (s)"]) < thr_sep) and \
                (idx < ripples.shape[0] - 1):
            idx += 1

        out.append(_)
        pbar.update(1)

    _ = pd.concat(out, axis=1).T
    _.columns.name = ""

    return _.reindex(sorted(_.columns), axis=1)


def get_trajectory_across_time_space_by_duration(session_id, ripples, spatial_info, source_area, position):
    """
    Divide strong and common ripples
    """

    #print(f"Processing session {session_id}-source area {source_area} ")

    real_ripple_summary = find_ripples_clusters_new(ripples, source_area)

    columns_to_keep = real_ripple_summary.loc[:, real_ripple_summary.columns.str.contains('lag')].columns
    proben_area = [q.split(" ")[0] for q in columns_to_keep]
    probe_n = [q.split("-")[0] for q in proben_area]
    area = [q.split("-")[1] for q in proben_area]

    pos = []
    for p, a in zip(probe_n, area):
        pos.append(spatial_info[(spatial_info["Probe number"] == int(p)) & (spatial_info["Area"] == a)]["M-L (µm)"])

    lag_stronger = pd.concat([real_ripple_summary[
                                    real_ripple_summary["Duration (s)"] > real_ripple_summary["Duration (s)"].quantile(.9)][
                                    columns_to_keep].mean().reset_index(drop=True) * 1000,
                                pd.concat(pos).reset_index(drop=True)], axis=1)
    lag_stronger.columns = ["Lag (ms)", "M-L (µm)"]
    lag_stronger.sort_values(by="M-L (µm)", inplace=True)
    lag_stronger["Session"] = session_id
    lag_stronger["Probe number-area"] = proben_area
    lag_stronger["Type"] = "Strong ripples"
    lag_stronger["Location"] = position

    lag_weaker = pd.concat([real_ripple_summary[
                                  real_ripple_summary["Duration (s)"] < real_ripple_summary["Duration (s)"].quantile(.9)][
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

    #print(f"Processing session {session_id}-source area {source_area} done!")

    out = pd.concat([lag_stronger, lag_weaker, lag_tot]).reset_index(drop=True)

    real_ripple_summary["Strong"] = real_ripple_summary["Duration (s)"] > real_ripple_summary["Duration (s)"].quantile(.9)


    return out#, pd.concat([_, __, ___], axis=1).T

def get_trajectory_across_time_space_by_strength(session_id, ripples, spatial_info, source_area, position):
    """
    Divide strong and common ripples
    """

    #print(f"Processing session {session_id}-source area {source_area} ")

    real_ripple_summary = find_ripples_clusters_new(ripples, source_area)

    columns_to_keep = real_ripple_summary.loc[:, real_ripple_summary.columns.str.contains('lag')].columns
    proben_area = [q.split(" ")[0] for q in columns_to_keep]
    probe_n = [q.split("-")[0] for q in proben_area]
    area = [q.split("-")[1] for q in proben_area]

    pos = []
    for p, a in zip(probe_n, area):
        pos.append(spatial_info[(spatial_info["Probe number"] == int(p)) & (spatial_info["Area"] == a)][["M-L (µm)", "A-P (µm)", "D-V (µm)"]])

    lag_stronger = pd.concat([real_ripple_summary[
                                    real_ripple_summary["∫Ripple"] > real_ripple_summary["∫Ripple"].quantile(.9)][
                                    columns_to_keep].mean().reset_index(drop=True) * 1000,
                                pd.concat(pos).reset_index(drop=True)], axis=1)
    lag_stronger.columns = ["Lag (ms)", "M-L (µm)", "A-P (µm)", "D-V (µm)"]
    lag_stronger.sort_values(by="M-L (µm)", inplace=True)
    lag_stronger["Session"] = session_id
    lag_stronger["Probe number-area"] = proben_area
    lag_stronger["Type"] = "Strong ripples"
    lag_stronger["Location"] = position

    lag_weaker = pd.concat([real_ripple_summary[
                                  real_ripple_summary["∫Ripple"] < real_ripple_summary["∫Ripple"].quantile(.9)][
                                  columns_to_keep].mean().reset_index(drop=True) * 1000,
                              pd.concat(pos).reset_index(drop=True)], axis=1)
    lag_weaker.columns = ["Lag (ms)", "M-L (µm)", "A-P (µm)", "D-V (µm)"]
    lag_weaker.sort_values(by="M-L (µm)", inplace=True)
    lag_weaker["Session"] = session_id
    lag_weaker["Probe number-area"] = proben_area
    lag_weaker["Type"] = "Common ripples"
    lag_weaker["Location"] = position

    lag_tot = pd.concat([real_ripple_summary[
                             columns_to_keep].mean().reset_index(drop=True) * 1000,
                         pd.concat(pos).reset_index(drop=True)], axis=1)
    lag_tot.columns = ["Lag (ms)", "M-L (µm)", "A-P (µm)", "D-V (µm)"]
    lag_tot.sort_values(by="M-L (µm)", inplace=True)
    lag_tot["Session"] = session_id
    lag_tot["Probe number-area"] = proben_area
    lag_tot["Type"] = "Total ripples"
    lag_tot["Location"] = position

    #print(f"Processing session {session_id}-source area {source_area} done!")

    out = pd.concat([lag_stronger, lag_weaker, lag_tot]).reset_index(drop=True)

    real_ripple_summary["Strong"] = real_ripple_summary["∫Ripple"] > real_ripple_summary["∫Ripple"].quantile(.9)

    # _ = real_ripple_summary[real_ripple_summary["Strong"] == 1]["Direction"].value_counts() / \
    #     real_ripple_summary[real_ripple_summary["Strong"] == 1]["Direction"].dropna().shape[0]
    # _["Session id"] = session_id
    # _["Type"] = "Strong"
    # __ = real_ripple_summary[real_ripple_summary["Strong"] == 0]["Direction"].value_counts() / \
    #      real_ripple_summary[real_ripple_summary["Strong"] == 0]["Direction"].dropna().shape[0]
    # __["Session id"] = session_id
    # __["Type"] = "Common"
    # ___ = real_ripple_summary["Direction"].value_counts() / real_ripple_summary["Direction"].dropna().shape[0]
    # ___["Session id"] = session_id
    # ___["Type"] = "Total"

    return out#, pd.concat([_, __, ___], axis=1).T


def get_trajectory_across_time_space_by_seed(session_id, ripples, spatial_info, source_area, position):
    """
    Divide ripples with local seed and not local seed
    """

    #print(f"Processing session {session_id}-source area {source_area} ")

    real_ripple_summary = find_ripples_clusters_new(ripples, source_area)

    columns_to_keep = real_ripple_summary.loc[:, real_ripple_summary.columns.str.contains('lag')].columns
    proben_area = [q.split(" ")[0] for q in columns_to_keep]
    probe_n = [q.split("-")[0] for q in proben_area]
    area = [q.split("-")[1] for q in proben_area]

    _ = (real_ripple_summary[columns_to_keep] >= 0) | (real_ripple_summary[columns_to_keep].isna())
    idx = _.all(axis=1)
    real_ripple_summary["Is source"] = idx

    pos = []
    for p, a in zip(probe_n, area):
        pos.append(spatial_info[(spatial_info["Probe number"] == int(p)) & (spatial_info["Area"] == a)][["M-L (µm)", "A-P (µm)", "D-V (µm)"]])

    lag_local = pd.concat([real_ripple_summary[
                                    real_ripple_summary["Is source"]==True][
                                    columns_to_keep].mean().reset_index(drop=True) * 1000,
                                pd.concat(pos).reset_index(drop=True)], axis=1)
    lag_local.columns = ["Lag (ms)", "M-L (µm)", "A-P (µm)", "D-V (µm)"]
    lag_local.sort_values(by="M-L (µm)", inplace=True)
    lag_local["Session"] = session_id
    lag_local["Probe number-area"] = proben_area
    lag_local["Type"] = "Local"
    lag_local["Location"] = position

    lag_non_local = pd.concat([real_ripple_summary[
                                  real_ripple_summary["Is source"]==False][
                                  columns_to_keep].mean().reset_index(drop=True) * 1000,
                              pd.concat(pos).reset_index(drop=True)], axis=1)
    lag_non_local.columns = ["Lag (ms)", "M-L (µm)", "A-P (µm)", "D-V (µm)"]
    lag_non_local.sort_values(by="M-L (µm)", inplace=True)
    lag_non_local["Session"] = session_id
    lag_non_local["Probe number-area"] = proben_area
    lag_non_local["Type"] = "Non-local"
    lag_non_local["Location"] = position

    lag_tot = pd.concat([real_ripple_summary[
                             columns_to_keep].mean().reset_index(drop=True) * 1000,
                         pd.concat(pos).reset_index(drop=True)], axis=1)
    lag_tot.columns = ["Lag (ms)","M-L (µm)", "A-P (µm)", "D-V (µm)"]
    lag_tot.sort_values(by="M-L (µm)", inplace=True)
    lag_tot["Session"] = session_id
    lag_tot["Probe number-area"] = proben_area
    lag_tot["Type"] = "Total ripples"
    lag_tot["Location"] = position

    #print(f"Processing session {session_id}-source area {source_area} done!")

    out = pd.concat([lag_local, lag_non_local, lag_tot]).reset_index(drop=True)

    # _ = real_ripple_summary[real_ripple_summary["Is source"] == "Source"]["Direction"].value_counts() / \
    #     real_ripple_summary[real_ripple_summary["Is source"] == "Source"]["Direction"].dropna().shape[0]
    # _["Session id"] = session_id
    # _["Type"] = "Source"
    # __ = real_ripple_summary[real_ripple_summary["Is source"] == "Secondary"]["Direction"].value_counts() / \
    #      real_ripple_summary[real_ripple_summary["Is source"] == "Secondary"]["Direction"].dropna().shape[0]
    # __["Session id"] = session_id
    # __["Type"] = "Secondary"
    # ___ = real_ripple_summary["Direction"].value_counts() / real_ripple_summary["Direction"].dropna().shape[0]
    # ___["Session id"] = session_id
    # ___["Type"] = "Total"

    return out#, pd.concat([_, __, ___], axis=1).T

def get_direction_no_spatial_division(session_id, ripples):

    real_ripple_summary = find_ripples_clusters_new_total(ripples, session_id)
    _ = real_ripple_summary["Direction"].value_counts() / real_ripple_summary["Direction"].dropna().shape[0]
    _["Session id"] = session_id
    return _

def upsample(trace):
    out = []
    out_2 = []
    idx = list(trace.index)
    for q in range(len(idx)):
        out.append(idx[q])
        if q + 1 < len(idx):
            out.append(((idx[q] + idx[q+1])/2))
        out_2.append(trace.iloc[q])
        if q + 1 < len(idx):
            out_2.append(np.nan)
    trace_new = pd.Series(out_2, index=out)

    trace_new = trace_new.loc[trace_new.first_valid_index(): trace_new.last_valid_index()]

    trace_new = trace_new.interpolate(method="pchip")

    return trace_new


def interpolate_and_reindex(trace):

    for q in range(5):
        trace = upsample(trace)

    idx = list(trace.index)
    new_idx = range(800, 4800, 20) # I cut the borders so it's ok if it is overboard
    idx.extend(new_idx)
    idx = list(set(idx))
    idx.sort()

    trace_new = trace.reindex(idx)

    trace_new = trace_new.loc[trace_new.first_valid_index() : trace_new.last_valid_index()]

    trace_nans = trace_new[trace_new.isna()]

    trace_nans[trace_nans.index.get_loc(trace_new[trace_new == 0].index[0], method="nearest")] = 0  # zero remains zero

    trace_new = pd.concat([trace_nans, trace_new[~trace_new.isna()]]).sort_index()

    trace_new = trace_new.interpolate(method="pchip")

    trace_new = trace_new.reindex(new_idx)

    trace_new = trace_new.loc[trace_new.first_valid_index() : trace_new.last_valid_index()]
    
    return trace_new

def batch_trajectories(ripples_calcs, kind, func):
    """
    spatial_info[0]= relative to source/seed
    spatial_info[1]= relative to all others CA1 locations
    """

    print(f"var_thr: {var_thr}")
    input_rip = []
    for session_id in ripples_calcs.keys():
        ripples = ripples_calcs[session_id][3].copy()
        ripples = ripples[ripples["Area"] == "CA1"]
        ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)
        input_rip.append(ripples.groupby("Probe number-area")["M-L (µm)"].mean())

    ml_space = pd.concat(input_rip)

    medial_lim = ml_space.quantile(.33333)
    lateral_lim = ml_space.quantile(.666666)
    center = ml_space.median()

    input_rip = []
    spatial_infos = []

    for session_id in tqdm(ripples_calcs.keys()):
        ripples = ripples_calcs[session_id][3].copy()

        sel_probe = ripples_calcs[session_id][5]

        if ripples[ripples['Probe number'] == sel_probe].shape[0] < 1000:
            continue

        ripples = ripples.sort_values(by="Start (s)").reset_index(drop=True)

        ripples = ripples[ripples["Area"] == "CA1"]

        ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)

        if ripples["Probe number-area"].unique().shape[0] < 2:
            continue

        ripples["Local strong"] = ripples.groupby("Probe number").apply(
            lambda x: x["∫Ripple"] > x["∫Ripple"].quantile(.9)).sort_index(level=1).values

        ripples["Session"] = session_id

        ripples = ripples.reset_index().rename(columns={'index': 'Ripple number'})

        ripples["Z-scored ∫Ripple"] = ripples.groupby("Probe number-area").apply(
            lambda group: zscore(group["∫Ripple"], ddof=1)).droplevel(0)

        spatial_info = ripples_calcs[session_id][1].copy()

        # print(c.groupby("Probe number-area").mean()["M-L (µm)"])
        if kind == "medial":
            position = "Medial"
            source_area = str(ripples["Probe number"].min()) + "-CA1"
            if (np.any(ripples.groupby("Probe number-area")["M-L (µm)"].mean() < medial_lim)) & (np.any(ripples.groupby("Probe number-area")["M-L (µm)"].mean() > medial_lim)): # check that there are recording sites in other hippocampal sections
                spatial_infos.append([ripples.groupby("Probe number-area").mean().loc[source_area, :], pd.DataFrame(
                    ripples.groupby("Probe number-area").mean().loc[
                    ripples.groupby("Probe number-area").mean().index != source_area, :])])
                input_rip.append((session_id, ripples, spatial_info, source_area, position))
        elif kind == "lateral":
            position = "Lateral"
            source_area = str(ripples["Probe number"].max()) + "-CA1"
            if (np.any(ripples.groupby("Probe number-area")["M-L (µm)"].mean() < lateral_lim)) & (np.any(ripples.groupby("Probe number-area")["M-L (µm)"].mean() > lateral_lim)):
                spatial_infos.append([ripples.groupby("Probe number-area").mean().loc[source_area, :], pd.DataFrame(
                                      ripples.groupby("Probe number-area").mean().loc[
                                      ripples.groupby("Probe number-area").mean().index != source_area, :])])
                input_rip.append((session_id, ripples, spatial_info, source_area, position))
        elif kind == "center":
            position = "Center"
            if (np.any(ripples.groupby("Probe number-area")["M-L (µm)"].mean().between(medial_lim, lateral_lim))) & (np.any(~ripples.groupby("Probe number-area")["M-L (µm)"].mean().between(medial_lim, lateral_lim))):
                source_area = ripples.groupby("Probe number-area")["M-L (µm)"].mean().sub(center).abs().idxmin()
                spatial_infos.append([ripples.groupby("Probe number-area").mean().loc[source_area, :], pd.DataFrame(
                    ripples.groupby("Probe number-area").mean().loc[
                    ripples.groupby("Probe number-area").mean().index != source_area, :])])
                input_rip.append((session_id, ripples, spatial_info, source_area, position))

    with Pool(processes=len(input_rip)) as pool:
        r = pool.starmap_async(func, input_rip)
        list_trajectories = r.get()
        pool.terminate()
    trajectories = pd.concat(list_trajectories).reset_index(drop=True)

    return trajectories, spatial_infos


def spike_plot(probes_list, axs, column, time_center, window, row_summary):

    color_palette = sns.color_palette("flare", 255)

    kde_plot_data = []
    colors = []
    for n in tqdm_notebook(range( row_summary)):
        try:
            _ = probes_list[n]
            lr_scaled = round(((_[0] - 6500) / (10000 - 6500)) * 255)
            axs[n, column].eventplot(_[1].values(), linelengths=5, color=color_palette[lr_scaled]);
            colors.append(color_palette[lr_scaled])
            axs[n, column].set_title(f"LR (µm): {str(round(_[0], 2))}", fontsize=8)
            axs[n, column].vlines(time_center, axs[n, column].get_ylim()[0], axs[n, column].get_ylim()[1], alpha=.5,
                                  color="k")
        except:
            pass

        axs[n, column].axis("off")
        ## keep only xaxis
        # axs[n, 0].axes.get_yaxis().set_visible(False)
        # axs[n, 0].set_frame_on(False)
        # xmin, xmax = axs[n, 0].get_xaxis().get_view_interval()
        # ymin, ymax = axs[n, 0].get_yaxis().get_view_interval()
        # axs[n, 0].add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2.5))

        axs[n, column].set_xlim([time_center - window[0], time_center + window[1]])
        kde_plot_data.append(list(chain(*_[1].values())))

    my_palette = sns.color_palette(colors)

    sns.kdeplot(data=kde_plot_data[:len(my_palette)], palette=my_palette, common_norm=True,
                ax=axs[row_summary, column], legend=False, bw_adjust=.2)
    axs[row_summary, column].vlines(time_center, axs[row_summary, column].get_ylim()[0],
                                         axs[row_summary, column].get_ylim()[1], alpha=.5, color="k")
    axs[row_summary, column].set_xlim([time_center - window[0], time_center + window[1]])

    return axs

def sub_dict_spikes(units, time_sub_spike_times, field_to_use_to_compare, target_area):

    sub_probes_list = []
    time_space_sub_spike_times = dict(zip(units[units[field_to_use_to_compare] == target_area].index,
                                          itemgetter(*units[units[field_to_use_to_compare] == target_area].index)(time_sub_spike_times)))
    for probe_id in units[units[field_to_use_to_compare] == target_area].sort_values("left_right_ccf_coordinate")["probe_id"].unique():
        lr = units[units[field_to_use_to_compare] == target_area].groupby("probe_id").mean().loc[probe_id]["left_right_ccf_coordinate"]
        sub_probes_list.append([lr, {cluster_id: spikes for cluster_id, spikes in time_space_sub_spike_times.items() if
                                    units.loc[cluster_id, :]["probe_id"] == probe_id}])
    return sub_probes_list


def process_spike_hists(time_center, space_sub_spike_times, probe_ids, lrs, units_probe_id, window):

    time_space_sub_spike_times = {
        cluster_id: spikes[(spikes > time_center - window[0]) & (spikes < time_center + window[1])] for
        cluster_id, spikes in space_sub_spike_times.items()}

    probes_list = []
    for probe_id in probe_ids:
        lr = lrs.loc[probe_id]
        probes_list.append(
            [lr, {cluster_id: spikes for cluster_id, spikes in time_space_sub_spike_times.items() if
                  units_probe_id.loc[cluster_id] == probe_id}])

    out = []
    for _ in probes_list:
        out.append(
            np.histogram(np.array(list(chain(*_[1].values()))) - time_center,
                         bins=np.arange(- window[0], window[1], .01))[0])

    return np.array(out)


def process_spikes_per_ripple(time_center, space_sub_spike_times, window):

    time_space_sub_spike_times = {
        cluster_id: spikes[(spikes > time_center - window[0]) & (spikes < time_center + window[1])] for
        cluster_id, spikes in space_sub_spike_times.items()}

    # out = {cluster_id: len(spikes) / ((window[1] + window[0]) * 100) for cluster_id, spikes in time_space_sub_spike_times.items()}

    out = []
    for cluster_id, spikes in time_space_sub_spike_times.items():
        out.append((cluster_id, len(spikes)/((window[1] + window[0]) * 100))) # per 10 ms
        #out.append((cluster_id, len(spikes)/(((window[1] + window[0]) * 1000)/10))) # per 10 ms
    return out

def extract_spikes_per_ripple(time_center, space_sub_spike_times, window):

    time_space_sub_spike_times = {
        cluster_id: spikes[(spikes > time_center - window[0]) & (spikes < time_center + window[1])] for
        cluster_id, spikes in space_sub_spike_times.items()}

    return time_space_sub_spike_times

def process_spike_and_neuron_numbers_per_ripple(time_space_sub_spike_times):

    #delte empties
    time_space_sub_spike_times = {i: j for i, j in time_space_sub_spike_times.items() if j.size > 0}

    if len(list(time_space_sub_spike_times.keys()))> 0:
        number_spikes = np.concatenate(list(time_space_sub_spike_times.values())).shape[0]
        number_participating_neurons = len(time_space_sub_spike_times.keys())
    else:
        number_spikes = 0
        number_participating_neurons = 0

    return number_spikes,  number_participating_neurons

def batch_process_spike_hists_by_seed_location(func, real_ripple_summary, units, spike_times, target_area,
                              field_to_use_to_compare, n_cpus, window):

    space_sub_spike_times = dict(zip(units[units[field_to_use_to_compare] == target_area].index,
                                     itemgetter(*units[units[field_to_use_to_compare] == target_area].index)(
                                         spike_times)))

    probe_ids = units[units[field_to_use_to_compare] == target_area].sort_values("left_right_ccf_coordinate")[
        "probe_id"].unique()
    lrs = units[units[field_to_use_to_compare] == target_area].groupby("probe_id").mean().sort_values("left_right_ccf_coordinate")[
        "left_right_ccf_coordinate"]
    #print(units[units[field_to_use_to_compare] == target_area].sort_values("left_right_ccf_coordinate")[
    #    "probe_id"].unique())
    #print(f"probe_ids:{probe_ids}", f"Check lr corresponds: {lrs}")
    units_probe_id = units["probe_id"]

    input_process_spike_hists = []

    for index, row in real_ripple_summary[real_ripple_summary["Location seed"] == "Medial"].iterrows():
        time_center = row["Start (s)"] + row[row.index.str.contains('lag')].min()# either sum zero or sum a negative value
        input_process_spike_hists.append\
            ((time_center, space_sub_spike_times, probe_ids, lrs, units_probe_id,  window))

    with Pool(processes=60) as pool:
        r = pool.starmap_async(func, input_process_spike_hists, chunksize=250)
        out_hist_medial = r.get()
        pool.close()

    input_process_spike_hists = []

    for index, row in real_ripple_summary[real_ripple_summary["Location seed"] == "Lateral"].iterrows():
        time_center = row["Start (s)"] + row[row.index.str.contains('lag')].min()
        input_process_spike_hists.append((time_center, space_sub_spike_times, probe_ids, lrs, units_probe_id, window))

    with Pool(processes=n_cpus) as pool:
        r = pool.starmap_async(func, input_process_spike_hists, chunksize=250)
        out_hist_lateral = r.get()
        pool.close()

    return lrs, out_hist_medial, out_hist_lateral

def batch_process_spike_hists_by_seed_location_and_strength(func, real_ripple_summary, units, spike_times, target_area,
                              field_to_use_to_compare, n_cpus, window):

    space_sub_spike_times = dict(zip(units[units[field_to_use_to_compare] == target_area].index,
                                     itemgetter(*units[units[field_to_use_to_compare] == target_area].index)(
                                         spike_times)))

    probe_ids = units[units[field_to_use_to_compare] == target_area].sort_values("left_right_ccf_coordinate")[
        "probe_id"].unique()
    lrs = units[units[field_to_use_to_compare] == target_area].groupby("probe_id").mean().sort_values("left_right_ccf_coordinate")[
        "left_right_ccf_coordinate"]
    #print(units[units[field_to_use_to_compare] == target_area].sort_values("left_right_ccf_coordinate")[
    #    "probe_id"].unique())
    #print(f"probe_ids:{probe_ids}", f"Check lr corresponds: {lrs}")
    units_probe_id = units["probe_id"]

    input_process_spike_hists = []

    for index, row in real_ripple_summary[real_ripple_summary["Local strong"] == True].iterrows():
        time_center = row["Start (s)"] + row[row.index.str.contains('lag')].min()# either sum zero or sum a negative value
        input_process_spike_hists.append\
            ((time_center, space_sub_spike_times, probe_ids, lrs, units_probe_id,  window))

    with Pool(processes=60) as pool:
        r = pool.starmap_async(func, input_process_spike_hists, chunksize=250)
        out_hist_strong = r.get()
        pool.close()

    input_process_spike_hists = []

    for index, row in real_ripple_summary[real_ripple_summary["Local strong"] == False].iterrows():
        time_center = row["Start (s)"] + row[row.index.str.contains('lag')].min()
        input_process_spike_hists.append((time_center, space_sub_spike_times, probe_ids, lrs, units_probe_id, window))

    with Pool(processes=n_cpus) as pool:
        r = pool.starmap_async(func, input_process_spike_hists, chunksize=250)
        out_hist_common = r.get()
        pool.close()

    return lrs, out_hist_strong, out_hist_common


def batch_process_spike_clusters_by_seed_location(func, real_ripple_summary, units, spike_times, target_area,
                              field_to_use_to_compare, n_cpus, window):

    if units[units[field_to_use_to_compare] == target_area].shape[0] > 0:

        space_sub_spike_times = dict(zip(units[units[field_to_use_to_compare] == target_area].index,
                                         itemgetter(*units[units[field_to_use_to_compare] == target_area].index)(
                                             spike_times)))

        lrs = units[units[field_to_use_to_compare] == target_area].groupby("probe_id").mean().sort_values("left_right_ccf_coordinate")["left_right_ccf_coordinate"]

        input_process_spike_hists = []
        duration_ripples_medial = []
        for index, row in real_ripple_summary[real_ripple_summary["Location seed"] == "Medial"].iterrows():
            time_center = row["Start (s)"] + row[row.index.str.contains('lag')].min() # either sum zero or sum a negative value
            duration_ripples_medial.append(row["Duration (s)"])
            input_process_spike_hists.append((time_center, space_sub_spike_times, window))

        with Pool(processes=n_cpus) as pool:
            r = pool.starmap_async(func, input_process_spike_hists, chunksize=250)
            out_medial = r.get()
            pool.close()

        input_process_spike_hists = []

        duration_ripples_lateral = []
        for index, row in real_ripple_summary[real_ripple_summary["Location seed"] == "Lateral"].iterrows():
            time_center = row["Start (s)"] + row[row.index.str.contains('lag')].min()
            duration_ripples_lateral.append(row["Duration (s)"])
            input_process_spike_hists.append((time_center, space_sub_spike_times, window))

        with Pool(processes=n_cpus) as pool:
            r = pool.starmap_async(func, input_process_spike_hists, chunksize=250)
            out_lateral = r.get()
            pool.close()

    return lrs, out_medial, out_lateral, duration_ripples_medial, duration_ripples_lateral, units


def plot_summary_spikes_hist(axs, n, out_hist, lrs):
    lr = lrs.iloc[n]
    color_palette = sns.color_palette("flare", 255)
    lr_scaled = round(((lr -6500) / (10000 - 6500)) * 255)
    y = np.array([_[n] for _ in out_hist]).mean(axis=0)
    x = np.arange(-.24, .25, .01)

    upsamped = scipy.interpolate.CubicSpline(x, y)
    y = upsamped(np.arange(-.24, .25, .001))
    x = np.arange(-.24, .25, .001) * 1000

    axs.plot(x, y, color=color_palette[lr_scaled])
    error = np.array([_[n] for _ in out_hist]).std(axis=0)/np.sqrt(np.array([_[n] for _ in out_hist]).shape[0])

    upsamped = scipy.interpolate.CubicSpline(np.arange(-.24, .25, .01), error)
    error = upsamped(np.arange(-.24, .25, .001))

    axs.fill_between(x, y-error, y+error, alpha=.3, color=color_palette[lr_scaled])

    #threshold_start = np.mean(y[:200]) + np.std(y[:200]) * 5
    threshold_start = 10

    #axs.scatter(x[np.argmax(y > threshold_start)], y[np.argmax(y > threshold_start)], s=40, facecolors='none',  edgecolors=color_palette[lr_scaled])
    axs.scatter(x[np.argmax(y)], y[np.argmax(y )], s=40, facecolors='none',
                edgecolors=color_palette[lr_scaled])

    #return [x[np.argmax(y > threshold_start)], y[np.argmax(y > threshold_start)], color_palette[lr_scaled]]
    return [x[np.argmax(y )], y[np.argmax(y)], color_palette[lr_scaled]]


def spike_per_ripple(space_sub_spike_times, ripples_starts):

    window_ripple = (0, 0.15)
    window_pre_ripple = (0.15, 0)
    out_s = []
    for time_center in tqdm_notebook(ripples_starts):

        time_space_sub_spike_times = {
            cluster_id: spikes[(spikes > time_center - window_ripple[0]) & (spikes < time_center + window_ripple[1])]
            for
            cluster_id, spikes in space_sub_spike_times.items()}

        # time_space_sub_spike_times_pre_ripple = {
        # cluster_id: spikes[(spikes > time_center - window_pre_ripple[0]) & (spikes < time_center + window_pre_ripple[1])] for
        # cluster_id, spikes in space_sub_spike_times.items()}

        out = []
        # for ripple_spikes, pre_ripple_spikes in zip(time_space_sub_spike_times.items(), time_space_sub_spike_times_pre_ripple.items()):
        for unit_id, spikes in time_space_sub_spike_times.items():
            # unit_id, spikes = ripple_spikes
            # unit_id, pre_spikes = pre_ripple_spikes

            out.append(len(spikes))  # - len(pre_spikes)

        out_s.append(out)

    out = pd.DataFrame(out_s)
    return out


def plot_trajs(trajectories, axs, ylim):

    trajectories_medial = trajectories
    sns.lineplot(data=trajectories_medial[(trajectories_medial["Type"]=="Strong ripples")],
                 y="Lag (ms)", x="M-L  (µm)", hue="Session", legend=False,  ax=axs[0], palette=sns.color_palette("Reds", as_cmap=True))
    axs[0].set_ylim(ylim)

    sns.lineplot(data=trajectories_medial[(trajectories_medial["Type"]=="Common ripples")],
                 y="Lag (ms)", x="M-L  (µm)", hue="Session", legend=False,  ax=axs[1], palette=sns.color_palette("Blues", as_cmap=True))
    axs[1].set_ylim(ylim)


def plot_trajs_interp(trajectories, axs, ylim):
    tqdm.pandas()
    interp_trajs = trajectories.groupby(["Session", "Type"]).progress_apply(lambda group: interpolate_and_reindex(group.set_index("M-L (µm)")["Lag (ms)"]))
    interp_trajs = pd.DataFrame(interp_trajs).reset_index()

    interp_trajs.columns = ["Session", "Type", "M-L (µm)", "Lag (ms)"]

    sns.lineplot(data=interp_trajs[(interp_trajs["Type"] == "Strong ripples")],
                 y="Lag (ms)", x="M-L (µm)", hue="Session", legend=False,  ax=axs[0], palette=sns.color_palette("Reds", as_cmap=True))
    axs[0].set_ylim(ylim)

    sns.lineplot(data=interp_trajs[(interp_trajs["Type"] == "Common ripples")],
                 y="Lag (ms)", x="M-L (µm)", hue="Session", legend=False,  ax=axs[1], palette=sns.color_palette("Blues", as_cmap=True))
    axs[1].set_ylim(ylim)


def plot_trajs_interp_by_seed(trajectories, axs, ylim):
    tqdm.pandas()
    interp_trajs = trajectories.groupby(["Session", "Type"]).progress_apply(lambda group: interpolate_and_reindex(group.set_index("M-L (µm)")["Lag (ms)"]))
    interp_trajs = pd.DataFrame(interp_trajs).reset_index()

    interp_trajs.columns = ["Session", "Type", "M-L (µm)", "Lag (ms)"]

    sns.lineplot(data=interp_trajs[(interp_trajs["Type"] == "Local")],
                 y="Lag (ms)", x="M-L (µm)", hue="Session", legend=False,  ax=axs[0], palette=sns.color_palette("light:#00A676", as_cmap=True))
    axs[0].set_ylim(ylim)

    sns.lineplot(data=interp_trajs[(interp_trajs["Type"] == "Non-local")],
                 y="Lag (ms)", x="M-L (µm)", hue="Session", legend=False,  ax=axs[1], palette=sns.color_palette("light:#6E2594", as_cmap=True))
    axs[1].set_ylim(ylim)


def spike_summary(spike_hists, area_to_analyze, reference_location, medial_lim, lateral_lim):
    out_common = []
    out_strong = []
    lrs_scaled_summary = []
    lrs_summary = []
    for session_id in [q[0] for q in list(spike_hists.keys()) if q[1] == area_to_analyze and q[2] == reference_location]:
        lrs, out_hist_weak, out_hist_strong = spike_hists[session_id, area_to_analyze, reference_location]
        lrs = lrs.fillna(value=10000 - 1000)
        lrs_scaled = round(((lrs - 6500) / (10000 - 6500)) * 255)

        for n in range(len(out_hist_weak[0])):
            y = np.array([_[n] for _ in out_hist_weak]).mean(axis=0)

            upsampled = scipy.interpolate.CubicSpline(np.arange(-.24, .25, .01), y)
            threshold_start = np.mean(upsampled(np.arange(-.24, .25, .0001))[:2000]) + \
                              np.std(upsampled(np.arange(-.24, .25, .0001))[:2000]) * 8

            out_common.append((np.arange(-.24, .25, .0001)[np.argmax(upsampled(np.arange(-.24, .25, .0001)) > threshold_start)] * 1000,
                        np.max(upsampled(np.arange(-.24, .25, .0001))), lrs_scaled.iloc[n]))

        for n in range(len(out_hist_strong[0])):
            y = np.array([_[n] for _ in out_hist_strong]).mean(axis=0)
            upsampled = scipy.interpolate.CubicSpline(np.arange(-.24, .25, .01), y)
            threshold_start = np.mean(upsampled(np.arange(-.24, .25, .0001))[:2000]) + \
                              np.std(upsampled(np.arange(-.24, .25, .0001))[:2000]) * 8

            out_strong.append((np.arange(-.24, .25, .0001)[np.argmax(upsampled(np.arange(-.24, .25, .0001)) > threshold_start)] * 1000,
                               np.max(upsampled(np.arange(-.24, .25, .0001))), lrs_scaled.iloc[n]))
        lrs_scaled_summary.append(lrs_scaled)
        lrs_summary.append(lrs)

    summary_weak = pd.DataFrame([np.array(out_common)[:, 0], np.array(out_common)[:, 1], pd.concat(lrs_summary)],
                                index=["Time from ripple start (ms)", "Spikes per 10 ms", "L-R (µm)"]).T
    summary_strong = pd.DataFrame([np.array(out_strong)[:, 0], np.array(out_strong)[:, 1], pd.concat(lrs_summary)],
                                  index=["Time from ripple start (ms)", "Spikes per 10 ms", "L-R (µm)"]).T

    def lr_classifier(row):
        if row["L-R (µm)"] < medial_lim:
            v = "Medial"
        elif row["L-R (µm)"] > lateral_lim:
            v = "Lateral"
        else:
            v = "Central"
        return v

    summary_weak["Location"] = summary_weak.apply(lr_classifier, axis=1)
    summary_weak["Type"] = "Common"
    summary_strong["Location"] = summary_strong.apply(lr_classifier, axis=1)
    summary_strong["Type"] = "Strong"
    summary = pd.concat([summary_weak, summary_strong])

    summary = summary[summary["Time from ripple start (ms)"].between(0, 100)]
    return summary


def format_for_annotator(out_test, field, between):
    out_test[field] = between
    try:
        out_test = out_test[out_test["p-unc"] < .05]
    except:
        try:
            out_test = out_test[out_test["p-tukey"] < .05]
        except:
            pass
    pair_A = tuple(zip(out_test[field], out_test["A"]))
    pair_B = tuple(zip(out_test[field], out_test["B"]))
    pairs = list(zip(pair_A, pair_B))
    try:
        pvalues = out_test["p-unc"].values
    except:
        try:
            pvalues = out_test["p-tukey"].values
        except:
            pass
    return pairs, pvalues

def postprocess_spike_hists(out_hist_lateral, out_hist_medial, lrs):
    means_l = []
    for n in range(len(out_hist_lateral[1])):
        means_l.append(np.array([_[n] for _ in out_hist_lateral]).mean(axis=0))

    means_m = []
    for n in range(len(out_hist_medial[1])):
        means_m.append(np.array([_[n] for _ in out_hist_medial]).mean(axis=0))

    mls = lrs - 5691.510009765625
    means = np.dstack([np.array(means_l), np.array(means_m)])
    means = xr.DataArray(means, coords=[("ML", mls), ("Time_ms", range(-240, 250, 10)),
                                        ("Seed", ["Lateral", "Medial"])])
    means_cut = means.sel(Time_ms=slice(-55, 130))
    return means_cut

def postprocess_spike_hists_strength(out_hist_common, out_hist_strong, lrs):
    means_strong = []
    for n in range(len(out_hist_strong[0])):
        means_strong.append(np.array([_[n] for _ in out_hist_strong]).mean(axis=0))

    means_common = []
    for n in range(len(out_hist_common[0])):
        means_common.append(np.array([_[n] for _ in out_hist_common]).mean(axis=0))

    mls = lrs - 5691.510009765625
    means = np.dstack([np.array(means_strong), np.array(means_common)])
    means = xr.DataArray(means, coords=[("ML", mls), ("Time_ms", range(-240, 250, 10)),
                                        ("Strength", ["Strong", "Common"])])
    means_cut = means.sel(Time_ms=slice(-55, 130))
    return means_cut

def plot_spike_hists_per_ML(means_cut):

    ml_space = get_ML_limits(var_thr)
    color_palette = sns.color_palette("flare", 255)

    fig, axs = plt.subplots(len(means_cut["ML"]), figsize=(4, 8))
    means_spiking = []
    for n, ML in enumerate(means_cut["ML"].values):
        # print(ML)

        _ = means_cut.sel(ML=ML).to_dataframe(name="Spiking per 10 ms").reset_index()
        _.rename(columns={"Time_ms": "Time (ms)"}, inplace=True)

        out_interp = []
        for seed in _["Seed"].unique():
            temp = _[_["Seed"] == seed].reset_index(drop=True)
            temp.index = range(1, 2 * len(temp) + 1, 2)
            temp = temp.reindex(index=range(2 * len(temp)))

            temp = temp.interpolate(method="cubicspline")
            temp["Seed"] = seed
            temp = temp.dropna()
            temp.index = range(1, 2 * len(temp) + 1, 2)
            temp = temp.reindex(index=range(2 * len(temp)))

            temp = temp.interpolate(method="cubicspline")
            temp["Seed"] = seed
            temp = temp.dropna()
            out_interp.append(temp)

        _ = pd.concat(out_interp).reset_index(drop=True)
        _ = _[_["Time (ms)"].between(-50, 125)]

        sns.lineplot(ax=axs[n], data=_, x="Time (ms)", y="Spiking per 10 ms", hue="Seed", palette=palette_ML, alpha=.2)
        bar = AnchoredSizeBar(axs[n].transData, 0, label='', size_vertical=5, loc="upper right", borderpad=1.4,
                              frameon=False)
        axs[n].add_artist(bar)

        seed = "Lateral"
        axs[n].fill_between(_[_["Seed"] == seed]["Time (ms)"], 0, _[_["Seed"] == seed]["Spiking per 10 ms"],
                            color=palette_ML[seed], alpha=.5)
        seed = "Medial"
        axs[n].fill_between(_[_["Seed"] == seed]["Time (ms)"], 0, _[_["Seed"] == seed]["Spiking per 10 ms"],
                            color=palette_ML[seed], alpha=.5)
        axs[n].vlines([np.arange(0, 125, 25)], axs[n].get_ylim()[0], axs[n].get_ylim()[1],
                      linestyle="--",
                      color="k", alpha=.5)

        if n < len(means_cut["ML"]) - 1:
            axs[n].axis("off")
            axs[n].get_legend().remove()

        else:
            axs[n].axis("on")
            axs[n].axes.get_yaxis().set_visible(False)
            axs[n].set_frame_on(False)
            xmin, xmax = axs[n].get_xaxis().get_view_interval()
            ymin, ymax = axs[n].get_yaxis().get_view_interval()
            axs[n].add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=3))
            axs[n].set_xlabel("Time from ripple start (ms)")

        axs[n].set_xlim([-50,125])

        _[_["Time (ms)"].between(0, 100)].groupby("Seed")["Spiking per 10 ms"].sum()
        means_spiking.append(_[_["Time (ms)"].between(0, 100)].groupby("Seed")["Spiking per 10 ms"].mean())

        axs[n].annotate(f"At {round(ML)} µm M-L", xy=(.6, .98), xycoords=axs[n].transAxes,
                        color=color_palette[round((ML-ml_space.min())/(ml_space.max()-ml_space.min())*255)], fontsize=6)

        print(_[_["Time (ms)"].between(0, 100)].groupby("Seed")["Spiking per 10 ms"].mean())


def spike_hists_diff_heatmap(means_cut):
    _ = pd.DataFrame(np.squeeze(means_cut.diff("Seed").to_numpy()), columns=means_cut["Time_ms"].astype(int),
                     index=means_cut["ML"])
    _.columns.name = "Time from ripple start (ms)"
    _.index.name = "M-L (µm)"
    sns.heatmap(data=_, cmap="seismic", cbar_kws={'label': 'Δ spiking per 10 ms'}, vmin=-10, vmax=10, xticklabels=5)


def get_ML_limits(var_thr):
    with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
        ripples_calcs = dill.load(f)

    input_rip = []
    for session_id in ripples_calcs.keys():
        ripples = ripples_calcs[session_id][3].copy()
        ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)
        input_rip.append(ripples.groupby("Probe number-area").mean()["M-L (µm)"])

    ml_space = pd.concat(input_rip)
    return ml_space

def Naturize():
    """
    Change figure panels lettering  to lowercase
    """
    for label in plt.figure(1).texts:
        if len(label.get_text())==1:
            label.set_text(label.get_text().lower())

def Naturize_text(legends_supplementary):
    """
    Change figure references to lowercase
    """
    for k, v in legends_supplementary.items():
        v_list = list(v)
        for n, char in enumerate(v_list):
            try:
                if (char.isupper()) & (v[n-1] == "(") & (v[n+1] == ")") & (char.isalpha() is True):
                    v_list[n] = char.lower()
                elif (char.isupper()) & (v[n-1] != ".") & (v[n-2] != ".") & (char.isalpha() is True) &\
                        (v[n-1].isalpha() is False) & (v[n+1].isalpha() is False)& (v[n+1]!="²"):
                    v_list[n] = char.lower()
                elif (char.isupper()) & (v[n-1].isdigit() is True):
                    v_list[n] = char.lower()
            except:
                pass
        legends_supplementary[k] = "".join(v_list)
    return legends_supplementary