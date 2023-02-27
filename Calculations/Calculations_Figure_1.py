import dill
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from Utils.Utils import calculate_lags, clean_ripples_calculations, find_couples_based_on_distance, calculate_corrs
import pickle
from Utils.Settings import var_thr, root_github_repo, output_folder_figures_calculations, output_folder_calculations, neuropixel_dataset

manifest_path = f"{neuropixel_dataset}/manifest.json"
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()

with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)


summary_corrs = calculate_corrs(ripples_calcs, sessions, var_thr)

high_distance, low_distance, distance_tabs = find_couples_based_on_distance(summary_corrs, ripples_calcs, sessions, var_thr)


invert_reference = False
ripples_lags = calculate_lags(high_distance, low_distance, sessions, invert_reference)

invert_reference = True
ripples_lags_inverted_reference = calculate_lags(high_distance, low_distance, sessions, invert_reference)

with open(f"{output_folder_figures_calculations}/temp_data_figure_1.pkl", "wb") as fp:
    pickle.dump([sessions, high_distance, low_distance, ripples_lags, ripples_lags_inverted_reference, ripples_calcs, summary_corrs, distance_tabs], fp)

from Figures.Figure_1.Util_Figure_1 import *

# create brainrenders
exec(open(f"{root_github_repo}/Figures/Figure_1/Figure_1_brainspace_high_distance.py").read())
exec(open(f"{root_github_repo}/Figures/Figure_1/Figure_1_brainspace_low_distance.py").read())
