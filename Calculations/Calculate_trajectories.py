import dill
from Utils.Utils import batch_trajectories, get_trajectory_across_time_space_by_strength, get_trajectory_across_time_space_by_seed
import pandas as pd
from Utils.Settings import output_folder_calculations

"""
Trajectories are called lag maps in the main text
"""

with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)


func = get_trajectory_across_time_space_by_strength

trajectories_medial, spatial_info_medial = batch_trajectories(ripples_calcs, "medial", func)
trajectories_lateral, spatial_info_lateral = batch_trajectories(ripples_calcs, "lateral", func)
trajectories_center, spatial_info_center = batch_trajectories(ripples_calcs, "center", func)

with open(f"{output_folder_calculations}/trajectories_by_strength.pkl",
          "wb") as fp:
    dill.dump(pd.concat([trajectories_medial, trajectories_lateral, trajectories_center]), fp)

with open(f"{output_folder_calculations}/trajectories_spatial_infos.pkl",
          "wb") as fp:
    dill.dump([spatial_info_medial, spatial_info_lateral, spatial_info_center], fp)


func = get_trajectory_across_time_space_by_seed

trajectories_medial, spatial_info_medial = batch_trajectories(ripples_calcs, "medial", func)
trajectories_lateral, spatial_info_lateral = batch_trajectories(ripples_calcs, "lateral", func)
trajectories_center, spatial_info_center = batch_trajectories(ripples_calcs, "center", func)

with open(f"{output_folder_calculations}/trajectories_by_seed.pkl",
          "wb") as fp:
    dill.dump(pd.concat([trajectories_medial, trajectories_lateral, trajectories_center]), fp)
