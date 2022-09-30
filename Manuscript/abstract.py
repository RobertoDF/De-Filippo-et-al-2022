from Utils.Settings import output_folder_calculations
from Utils.Results_variables import r_strong, summary_table
import dill

with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)


abstract = "Hippocampal ripples are highly synchronous neural events critical for memory consolidation and retrieval. " \
           "A minority of strong ripples has been shown to be of particular importance in situations of increased memory demands. " \
           "The propagation dynamics of strong ripples inside the hippocampal formation are, however, still opaque. " \
           f"We analyzed ripple propagation within the hippocampal formation in a large open access dataset comprising " \
           f"{summary_table.groupby('Session')['Probe number'].unique().explode().shape[0]} Neuropixel recordings in " \
           f"{summary_table['Session'].unique().shape[0]} awake, head-fixed mice. " \
           "Surprisingly, strong ripples (top 10% in ripple strength) " \
           "propagate differentially depending on their generation point along the hippocampal longitudinal axis. " \
           "The septal hippocampal pole is able to generate" \
           " longer ripples that engage more neurons " \
           " and elicit spiking activity for an extended time along the longitudinal axis of the hippocampal formation. " \
           "Accordingly, a substantial portion of the variance in strong ripple duration" \
           f" (R² = {r_strong}) is explained by the ripple generation location on the longitudinal axis. " \
           "Our results are consistent with a possible distinctive role of the hippocampal septal pole in conditions of high memory demand. "

#%%
