import dill
import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from tqdm import tqdm
from Utils.Settings import output_folder_calculations, neuropixel_dataset, var_thr
from Utils.Utils import acronym_to_main_area

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

out = []
for session_id in tqdm(ripples_calcs.keys()):
    ripples = ripples_calcs[session_id][3].copy()

    sel_probe = ripples_calcs[session_id][5]

    if ripples[ripples['Probe number'] == sel_probe].shape[0] < 1000:
        continue

    if np.any(ripples["L-R (µm)"].unique() < medial_lim) & np.any(ripples["L-R (µm)"].unique() > lateral_lim):
        print(sum(ripples["L-R (µm)"].unique() < medial_lim), sum(ripples["L-R (µm)"].unique() > lateral_lim))
        session = cache.get_session_data(session_id,  amplitude_cutoff_maximum = np.inf, presence_ratio_minimum = -np.inf, isi_violations_maximum = np.inf)  # , amplitude_cutoff_maximum = np.inf, presence_ratio_minimum = -np.inf, isi_violations_maximum = np.inf)

        units = session.units
        units["parent area"] = units["ecephys_structure_acronym"].apply(lambda area: acronym_to_main_area(area))
        units["session_id"] = session_id
        out.append(units[units["parent area"]=="HPF"])


total_units = pd.concat(out)

def l_r_classifier(row):
    if row["left_right_ccf_coordinate"] < medial_lim:
        v = "Medial"
    elif row["left_right_ccf_coordinate"] > lateral_lim:
        v = "Lateral"
    else:
        v = "Central"
    return v

total_units["Location"] = total_units.apply(l_r_classifier, axis=1)

def neuron_classifier(row):
    if row["waveform_duration"] < .4:
        v = "Putative inh"
    elif row["waveform_duration"] >= .4:
        v = "Putative exc"
    return v

total_units["Neuron type"] = total_units.apply(neuron_classifier, axis=1)

with open(f"{output_folder_calculations}/clusters_features_per_section.pkl", "wb") as fp:
    dill.dump(total_units, fp)