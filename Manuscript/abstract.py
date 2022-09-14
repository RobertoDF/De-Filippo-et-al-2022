from Utils.Settings import output_folder_calculations
from Utils.Results_variables import r_strong
import dill

with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)



abstract = "Hippocampal ripples are highly synchronous neural events critical for memory consolidation and retrieval. " \
           "A minority of strong ripples has been shown to be of particular importance in situations of increased memory demands. " \
           "The propagation dynamics of strong ripples inside the hippocampal formation are however still opaque. " \
           f"We analyzed extensively ripple propagation within the septal half of the hippocampal formation in an open access dataset" \
           f" provided by the Allen Institute. " \
           "Surprisingly, strong ripples propagate differentially in the septal and temporal direction along the hippocampal longitudinal axis. The majority " \
           "of strong ripples is always generated locally, however, the septal hippocampal pole is able to generate" \
           " longer ripples that engage more neurons" \
           " and elicit spiking activity for an extended time along the entire septal half of the hippocampal formation. Septally-generated " \
           "ripples have therefore higher chances of retaining their strength while travelling within the hippocampus. " \
           f"A substantial portion of the variance in strong ripple duration (R² = {r_strong}) is explained solely by the ripple starting " \
           f"position on the longitudinal axis.  " \
           "Our results suggest a possible distinctive role of the hippocampal septal pole in conditions of high memory demands. "
