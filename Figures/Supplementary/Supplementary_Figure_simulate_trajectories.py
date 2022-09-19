from Utils.Settings import output_folder_calculations, var_thr, Adapt_for_Nature_style
from Utils.Utils import Naturize
import dill
from Utils.Utils import format_for_annotator
import Utils.Style
from Utils.Style import palette_ML, palette_timelags
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statannotations.Annotator import Annotator
import pingouin as pg

# WIP

with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)

input_rip = []
for session_id in ripples_calcs.keys():
    ripples = ripples_calcs[session_id][3].copy()
    ripples = ripples[ripples["Area"] == "CA1"]
    ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)
    input_rip.append(ripples.groupby("Probe number-area").mean()["M-L (µm)"])

ml_space = pd.concat(input_rip)

medial_center_ml = ml_space.quantile(.33333/2)
lateral_center_ml = ml_space.quantile(.666666 + .33333/2)
center_center_ml = ml_space.quantile(.5)

speed_µm_ms = 2000/20

with open(f'{output_folder_calculations}/ripples_features_sessions_across_all_ML.pkl', 'rb') as f:
    out = dill.load(f)

seed_strong_ripples_by_hip_section_summary = pd.concat([q[6] for q in out])
seed_strong_ripples_by_hip_section_summary = seed_strong_ripples_by_hip_section_summary.reset_index().rename(columns={'index': 'Location seed'})

seed_common_ripples_by_hip_section_summary = pd.concat([q[3] for q in out])
seed_common_ripples_by_hip_section_summary = seed_common_ripples_by_hip_section_summary.reset_index().rename(columns={'index': 'Location seed'})


seed_common_ripples_by_hip_section_summary["Type"] = "Common ripples"
seed_strong_ripples_by_hip_section_summary["Type"] = "Strong ripples"
seed_ripples_by_hip_section_summary = pd.concat([seed_common_ripples_by_hip_section_summary, seed_strong_ripples_by_hip_section_summary])


profile = seed_ripples_by_hip_section_summary[seed_ripples_by_hip_section_summary["Reference"]
                                              =="Lateral"].groupby(["Type","Location seed"]).mean()\
                                                ["Percentage seed (%)"]["Strong ripples"]

summary = []

for seed, n in profile.iteritems():
    n = round(n)
    _ = []
    if seed == "Central seed":
        print(f"Central seed {n}")
        for q in range(n):
            _.append([0, (medial_center_ml - center_center_ml) / speed_µm_ms, (lateral_center_ml - center_center_ml) / speed_µm_ms + (medial_center_ml - center_center_ml) / speed_µm_ms] )
        summary.extend(_)

    elif seed == "Medial seed":
        print(f"Medial seed {n}")
        for q in range(n):
            _.append([0, -(medial_center_ml - center_center_ml) / speed_µm_ms, (lateral_center_ml - medial_center_ml) / speed_µm_ms])

        summary.extend(_)

    elif seed == "Lateral seed":
        print(f"Lateral seed {n}")
        for q in range(n):
            _.append([0, (medial_center_ml - center_center_ml) / speed_µm_ms, -(lateral_center_ml - medial_center_ml) / speed_µm_ms ])

        summary.extend(_)

np.mean(summary, axis=0)