from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import pandas as pd
import dill
import numpy as np
from Utils.Settings import minimum_firing_rate_hz, output_folder_figures_calculations, output_folder_calculations, neuropixel_dataset, var_thr, minimum_ripples_count_spike_analysis, minimum_ripples_count_generated_in_lateral_or_medial_spike_analysis

manifest_path = f"{neuropixel_dataset}/manifest.json"

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()
all_areas_recorded = [item for sublist in  sessions["ecephys_structure_acronyms"].to_list() for item in sublist]

count_areas_recorded = pd.Series(all_areas_recorded).value_counts()

with open(f'{output_folder_calculations}/units_summary_with_added_metrics.pkl', 'rb') as f:
    spikes_summary = dill.load(f)

summary_units_df = pd.concat(spikes_summary.values())

neurons_per_area = summary_units_df.groupby('ecephys_structure_acronym').size()

summary_units_df['Ripple modulation (0-50 ms) medial'] = (summary_units_df['Firing rate (0-50 ms) medial'] - summary_units_df['Firing rate (120-0 ms) medial'] )/ \
                                                            ( summary_units_df['Firing rate (120-0 ms) medial'])
summary_units_df['Ripple modulation (0-50 ms) lateral'] = (summary_units_df['Firing rate (0-50 ms) lateral'] - summary_units_df['Firing rate (120-0 ms) lateral']) / \
                                                            (summary_units_df['Firing rate (120-0 ms) lateral'])
summary_units_df['Ripple modulation (50-120 ms) medial'] = (summary_units_df['Firing rate (50-120 ms) medial']- summary_units_df['Firing rate (120-0 ms) medial']) / \
                                                               ( summary_units_df['Firing rate (120-0 ms) medial'])
summary_units_df['Ripple modulation (50-120 ms) lateral'] = (summary_units_df['Firing rate (50-120 ms) lateral']- summary_units_df['Firing rate (120-0 ms) lateral']) / \
                                                               ( summary_units_df['Firing rate (120-0 ms) lateral'])
summary_units_df['Ripple modulation (0-120 ms) medial'] = (summary_units_df['Firing rate (0-120 ms) medial'] - summary_units_df['Firing rate (120-0 ms) medial'])/ \
                                                            (summary_units_df['Firing rate (120-0 ms) medial'] )
summary_units_df['Ripple modulation (0-120 ms) lateral'] = (summary_units_df['Firing rate (0-120 ms) lateral']-summary_units_df['Firing rate (120-0 ms) lateral'])/\
                                                            ( summary_units_df['Firing rate (120-0 ms) lateral'])

summary_units_df['Pre-ripple modulation medial'] = (summary_units_df['Firing rate (20-0 ms) medial'] - summary_units_df['Firing rate (120-20 ms) medial'] ) / \
                                                (summary_units_df['Firing rate (120-20 ms) medial'] )
summary_units_df['Pre-ripple modulation lateral'] = (summary_units_df['Firing rate (20-0 ms) lateral']-summary_units_df['Firing rate (120-20 ms) lateral']) /  \
                                            (summary_units_df['Firing rate (120-20 ms) lateral'])


summary_units_df_sub = summary_units_df[(summary_units_df['ecephys_structure_acronym'].isin(count_areas_recorded[count_areas_recorded>8].index))&
                                        (summary_units_df['ecephys_structure_acronym'].isin(neurons_per_area[neurons_per_area>100].index))&
                                       (summary_units_df['ecephys_structure_acronym']!='grey')&
                                       (summary_units_df['ecephys_structure_acronym']!='HPF')]

summary_units_df_sub = summary_units_df_sub[~(summary_units_df_sub['Ripple modulation (0-50 ms) medial'].isin([np.nan, np.inf, -np.inf])) &
                        ~(summary_units_df_sub['Ripple modulation (0-50 ms) lateral'].isin([np.nan, np.inf, -np.inf])) &
                        ~(summary_units_df_sub['Ripple modulation (50-120 ms) medial'].isin([np.nan, np.inf, -np.inf])) &
                        ~(summary_units_df_sub['Ripple modulation (50-120 ms) lateral'].isin([np.nan, np.inf, -np.inf])) &
                        ~(summary_units_df_sub['Ripple modulation (0-120 ms) medial'].isin([np.nan, np.inf, -np.inf])) &
                        ~(summary_units_df_sub['Ripple modulation (0-120 ms) lateral'].isin([np.nan, np.inf, -np.inf])) &
                        ~(summary_units_df_sub['Pre-ripple modulation medial'].isin([np.nan, np.inf, -np.inf]))&
                        ~(summary_units_df_sub['Pre-ripple modulation lateral'].isin([np.nan, np.inf, -np.inf]))]


summary_units_df_sub.columns = summary_units_df_sub.columns.str.replace('_', ' ')
summary_units_df_sub.columns = summary_units_df_sub.columns.str.capitalize()

summary_units_df_sub = summary_units_df_sub.rename(columns={"Left right ccf coordinate":"L-R", "Anterior posterior ccf coordinate":"A-P", "Dorsal ventral ccf coordinate":"D-V"})
summary_units_df_sub = summary_units_df_sub.rename(columns={"Ecephys structure acronym":"Brain region", "Parent area":"Parent brain region"})

summary_units_df_sub["M-L"] = summary_units_df_sub["L-R"] - 5691.510009765625

def conditions_type_engagement(s):
    if (s['Ripple modulation (0-50 ms) medial'] > .5) | (s['Ripple modulation (0-50 ms) lateral'] > .5):
        if ((s['Ripple modulation (0-50 ms) medial']) / (s['Ripple modulation (0-50 ms) lateral'] + 1e-9)) > 2:
            return 'Medial ripple engagement'
        elif ((s['Ripple modulation (0-50 ms) lateral']) / (s['Ripple modulation (0-50 ms) medial'] + 1e-9)) > 2:
            return 'Lateral ripple engagement'
        else:
            return 'No preference'
    else:
        return 'No preference'


def conditions_ripple_engagement(s):
    if (s['Ripple modulation (0-50 ms) medial'] > .5) | ((s['Ripple modulation (0-50 ms) lateral']) > .5):
        return 'Ripple engagement'
    else:
        return 'No engagement'

summary_units_df_sub['Ripple type engagement'] = summary_units_df_sub[(summary_units_df_sub['Firing rate (0-50 ms) medial']>minimum_firing_rate_hz) |
                                                        (summary_units_df_sub['Firing rate (0-50 ms) lateral']>minimum_firing_rate_hz)].apply(conditions_type_engagement, axis=1)
summary_units_df_sub['Ripple engagement'] = summary_units_df_sub[(summary_units_df_sub['Firing rate (0-50 ms) medial']>minimum_firing_rate_hz) |
                                                        (summary_units_df_sub['Firing rate (0-50 ms) lateral']>minimum_firing_rate_hz)
                                                                 ].apply(conditions_ripple_engagement, axis=1)


summary_units_df_sub['Diff pre-ripple modulation (20-0 ms)'] = (summary_units_df_sub['Pre-ripple modulation medial'] - summary_units_df_sub['Pre-ripple modulation lateral'])
summary_units_df_sub['Diff ripple modulation (0-50 ms)'] = (summary_units_df_sub['Ripple modulation (0-50 ms) medial'] - summary_units_df_sub['Ripple modulation (0-50 ms) lateral'])
summary_units_df_sub['Diff ripple modulation (50-120 ms)'] = (summary_units_df_sub['Ripple modulation (50-120 ms) medial'] - summary_units_df_sub['Ripple modulation (50-120 ms) lateral'])
summary_units_df_sub['Diff firing rate (0-50 ms)'] = (summary_units_df_sub['Firing rate (0-50 ms) medial'] - summary_units_df_sub['Firing rate (0-50 ms) lateral'])
summary_units_df_sub['Diff firing rate (50-120 ms)'] = (summary_units_df_sub['Firing rate (50-120 ms) medial'] - summary_units_df_sub['Firing rate (50-120 ms) lateral'])

with open(f"{ output_folder_figures_calculations}/temp_data_figure_5.pkl", "wb") as fp:
    dill.dump(summary_units_df_sub, fp)

