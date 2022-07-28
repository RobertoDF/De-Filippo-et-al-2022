from Utils.Settings import output_folder_calculations
import dill

with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)



abstract = "Hippocampal ripples are highly synchronous neural events critical for memory consolidation and retrieval. " \
           "A minority of strong ripples has been shown to be of particular importance in situations of increased memory demands. " \
           "The propagation dynamics of strong ripples inside the hippocampal formation are however still opaque. " \
           f"Using an open access database of large-scale electrophysiological recordings in {len(ripples_calcs)} " \
           f"mice we extensively analyzed ripple propagation in the septal half of the hippocampal formation. " \
           "Strong ripples propagate differentially in the septal and temporal direction along the hippocampal longitudinal axis, " \
           "the hippocampus is therefore anisotropic in relation to ripple propagation. Hippocampal anisotropy is explained by " \
           "the ability of the septal hippocampal pole to generate" \
           " longer strong ripples that engage more neurons" \
           " and elicit spiking activity for an extended time along the entire septal half of the hippocampal formation."
