from Utils.Legends_supplementary_variables import *
from Utils.Settings import Adapt_for_Nature_style
from Utils.Utils import Naturize_text

legends_supplementary = {
    "Figure 1 - Figure supplement 1. Spatial coordinates of all recorded brain regions.": "2D histograms (upper diagonal), "
    "scatter plots (lower diagonal) and kernel density estimate plots (diagonal) of all the recorded regions color-coded according "
    "to the Allen Institute color scheme. HPF=hippocampus, TH=thalamus, HY=hypothalamus and MB=midbrain. M-L axis is zeroed at the midline.",

    "Figure 1 - Figure supplement 2. Correlation between ripple duration and strength per session.": "Red line represents linear regression with confidence interval of 95% estimated via bootstrap. "
                                                                                  "*** means p < 0.0005.",

    "Figure 1 - Figure supplement 3. Comparison between correlation of ripple strength and duration with underlying spiking.": "Ripple strength correlates significantly better with "
                                                                                                                       "the underlying ripple spiking activity. * means p < 0.0005. ",

    "Figure 1 - Figure supplement 4. Ripple-associated LFP responses are predominantly observed in hippocampal structures.":
   f"(A) Rendering of probe locatiosn for session {session_id_supp_fig1}. "
   f"(B) First column: Raw LFP traces color coded according to probe identity, "
   f"superimposed in black the trace after high-pass filtering to show the presence of a ripple. Scale bar: 250 µV. "
   f"Middle column: Ripple envelope and associated ∫Ripple in red. Last column: Raw LFP trace and associated RIVD"
   f" in blue. "
   f"(C) Heatmaps of ∫Ripple (left) and RIVD (right) for the entirety of session {session_id_supp_fig1} and "
   f"for each recorded area. To note the variability in ∫Ripple over time and cross different CA1 locations."
   f"(D) Kernel density estimate plot showing the relationship between ∫Ripple and RIVD. Bar plot shows the "
   f"sum of the z-scored ∫Ripple and RIVD per area."
   f"for the areas showing the strongest responses in session {session_id_supp_fig1}. "
   f"(E) Summary scatter plot showing the relationship between ∫Ripple and RIVD for all sessions. Bar plot shows the "
   f"sum of the z-scored ∫Ripple and RIVD per area averaged across animals. Most of the activity is confined to the hippocampal formation (DG, CA1, CA2, CA3 Sub and ProS)"
   f" (n={summary_table.Session.unique().shape[0]}). "
   f"(F) Violin plots showing the distribution of ∫Ripple and RIVD z-scored per session,"
   f" hippocampal regions (text in green) show the biggest responses.",

    "Figure 1 - Figure supplement 5. Hippocampal sections.": "(A) Variance explained between 3D distances and distance on each spatial axis across CA1 recording locations. "
                                                     "(B) Histogram showing the three sections across the M-L axis, the hippocampus was divided in order to have "
                                                      "an equal number of recordings in each section. "
                                                     "(C) Rendering of the 3 sections and associated recording locations (black dots).",

   "Figure 3 - Figure supplement 1. Spatio-temporal lag maps of locally and not locally generated ripples": ""
             "Spatio-temporal profiles are symmetrical, strong indication of similar propagation speed regardless of seed position. (A) Recording locations relative to (B). Red circles represents the reference locations across all sessions "
                        f"(n sessions={trajectories_by_strength['Session'].unique().shape[0]}), black circles represents the remaining recording locations. "
                        f"(B) Left: Medio-lateral propagation of locally generated ripples (generated in the reference section), each line represents the average of one session. "
                        f"Middle: Medio-lateral propagation of non-locally generated ripples, each line represents the average of one session. "
                        f"Right: Average propagation map across sessions of strong and common ripples. Reference locations are the most lateral per session. "
                         f"(C) Same as A. (D) Same as B. Reference locations are the most lateral per session. "
                        f"(E) Same as A. (F) Same as B. Reference locations are the most central per session. ",

   "Figure 3 - Figure supplement 2. Strength conservation in medially and laterally generated ripples.":
      "(A) Strength conservation index in strong ripples grouped by reference location. Ripples generated in the lateral section show"
      f"significantly lower strength conservation (p={round(pvalues_supp_4, 9)}, Student's t-test).  "
      "(B) Strength conservation index in common ripples grouped by reference location.",

   "Figure 3 - Figure supplement 3. Spatial location does not influence ∫Ripple.":
      "Relationship between Z-scored ∫Ripple (top row) or ∫Ripple (bottom row) and each spatial axis (M-L, A-P or D-V). Spatial location "
      f"has a negligible effect on ∫Ripple.",

   "Figure 3 - Figure supplement 4. Spatial location does not influence ripple amplitude.":
      "Relationship between Z-scored amplitude (top row) or amplitude (bottom row) and each spatial axis (M-L, A-P or D-V). Spatial location "
      f"has a negligible effect on ripple amplitude.",

   "Figure 4 - Figure supplement 1. Differential spiking of hippocampal neurons between different conditions.":
      "(A) Grand average of the differences between medial and lateral ripples induced spiking activity in "
      "putative excitatory (left) and inhibitory neurons (right). Putative excitatory and inhibitory neurons "
      "show similiar spiking patterns in lateral and medial ripples. "
      "(B) Grand average of the differences between common and strong ripples induced spiking activity in "
      "medial (left) and lateral ripples (right). Strong ripples are not associated with more spiking activity in the "
      "early phase post ripple start (0-50 ms)."
      "(C) Grand average of the differences between medial and lateral ripples induced spiking activity in "
      "common (left) and strong ripples (right). Strong ripples are associated with considerable differences between medial and "
      "lateral ripples."
    ,

   "Figure 4 - Figure supplement 2. Spiking rate and fraction of active neurons are significantly higher in medial ripples.":
      f"(A) Fraction of active neurons per ripple grouped by ripple seed location. (Medial seed={100*round(data_sup_9_fraction_clu.groupby('Location seed').mean()['Fraction active neurons per ripple (%)']['Medial seed'], 2)}±"
      f"{100*round(data_sup_9_fraction_clu.groupby('Location seed').sem()['Fraction active neurons per ripple (%)']['Medial seed'], 2)}%, lateral seed={100*round(data_sup_9_fraction_clu.groupby('Location seed').mean()['Fraction active neurons per ripple (%)']['Lateral seed'], 2)}±"
      f"{100*round(data_sup_9_fraction_clu.groupby('Location seed').sem()['Fraction active neurons per ripple (%)']['Lateral seed'], 2)}%, p-value={'{:.2e}'.format(ttest_clus_per_ripple['p-val'].values[0])}, Student's t-test). "
      f"(B) Average spiking rate grouped per ripple grouped by ripple seed location (Medial seed={100*round(data_sup_9_spiking_rate.groupby('Location seed').mean()['Spiking rate per 10 ms']['Medial seed'], 2)}±"
      f"{100*round(data_sup_9_spiking_rate.groupby('Location seed').sem()['Spiking rate per 10 ms']['Medial seed'], 2)}%, lateral seed={100*round(data_sup_9_spiking_rate.groupby('Location seed').mean()['Spiking rate per 10 ms']['Lateral seed'], 2)}±"
      f"{100*round(data_sup_9_spiking_rate.groupby('Location seed').sem()['Spiking rate per 10 ms']['Lateral seed'], 2)}%, p-value={'{:.2e}'.format(ttest_late_spiking['p-val'].values[0])}, Student's t-test). "
      f"Asterisks mean p < 0.05, Student's t-test.",

   "Figure 4 - Figure supplement 3. Spiking rate and fraction of active neurons are increased in the late phase post-ripple."
   "start in medial ripples both in putative excitatory and inhibitory neurons.":
      "(A) Average spiking rate in early (left) and late (right) phase post-ripple start grouped by ripple seed location and putative neuron identity. "
      "Asterisks mean p < 0.05, ANOVA with pairwise Tukey post-hoc test. "
      "(B) Fraction of active neurons per ripple in early (left) and late (right) phase post-ripple start grouped by ripple "
      "seed location and putative neuron identity. Asterisks mean p < 0.05, ANOVA with pairwise Tukey post-hoc test.",

"Figure 4 - Figure supplement 4. Units features in medial and lateral sections.":
      f"(A) Kernel density estimate plot of waveform duration (p-value={'{:.2e}'.format(p_val_wav_dur_inh)}), firing rate (p-value={'{:.2e}'.format(p_val_fir_rate_inh)}), "
      f"waveform amplitude (p-value={'{:.2e}'.format(p_val_wav_amp_inh)}), waveform repolarization slope (p-value={'{:.2e}'.format(p_val_wav_repolarization_slope_inh)}), "
      f"waveform recovery slope (p-value={'{:.2e}'.format(p_val_wav_rec_slope_inh)}) and waveform peak-through ratio (p-value={'{:.2e}'.format(p_val_wav_PT_ratio_inh)}) grouped by hippocampal section."
      "Asterisks mean p<0.05, Mann-Whitney U test. "
      f"(B) Cumulative distribution plot of waveform duration (p-value={'{:.2e}'.format(p_val_ks_wav_dur_inh)}), firing rate (p-value={'{:.2e}'.format(p_val_ks_fir_rate_inh)}), "
      f"waveform amplitude (p-value={'{:.2e}'.format(p_val_ks_wav_amp_inh)}), waveform repolarization slope (p-value={'{:.2e}'.format(p_val_ks_wav_repolarization_slope_inh)}), "
      f"waveform recovery slope (p-value={'{:.2e}'.format(p_val_ks_wav_rec_slope_inh)}) and waveform peak-through ratio (p-value={'{:.2e}'.format(p_val_ks_wav_PT_ratio_inh)}) grouped by hippocampal section."
      "Asterisks mean p < 0.05, Kolgomorov-Smirnov test."
      f"(C) Kernel density estimate plot of waveform duration (p-value={'{:.2e}'.format(p_val_wav_dur_exc)}), firing rate (p-value={'{:.2e}'.format(p_val_fir_rate_exc)}), "
      f"waveform amplitude (p-value={'{:.2e}'.format(p_val_wav_amp_exc)}), waveform repolarization slope (p-value={'{:.2e}'.format(p_val_wav_repolarization_slope_exc)}), "
      f"waveform recovery slope (p-value={'{:.2e}'.format(p_val_wav_rec_slope_exc)}) and waveform peak-through ratio (p-value={'{:.2e}'.format(p_val_wav_PT_ratio_exc)}) grouped by hippocampal section."
      "Asterisks mean p<0.05, Mann-Whitney U test. "
      f"(D) Cumulative distribution plot of waveform duration (p-value={'{:.2e}'.format(p_val_ks_wav_dur_exc)}), firing rate (p-value={'{:.2e}'.format(p_val_ks_fir_rate_exc)}), "
      f"waveform amplitude (p-value={'{:.2e}'.format(p_val_ks_wav_amp_exc)}), waveform repolarization slope (p-value={'{:.2e}'.format(p_val_ks_wav_repolarization_slope_exc)}), "
      f"waveform recovery slope (p-value={'{:.2e}'.format(p_val_ks_wav_rec_slope_exc)}) and waveform peak-through ratio (p-value={'{:.2e}'.format(p_val_ks_wav_PT_ratio_exc)}) grouped by hippocampal section."
      "Asterisks mean p < 0.05, Kolgomorov-Smirnov test.",

"Figure 5 - Figure supplement 1. Spiking rate modulation in medial and lateral ripples across brain regions.":
    "(A) Relationship between baseline (120 ms before ripple start) and medial ripple (0-50 ms) firing rate for clusters recorded in  HPF, Isocortex, MB and TH. In the Isocortex and MB plot "
    "we excluded the minority of clusters showing modulation >50% in response to either lateral or medial ripples(grey dots). Dashed black line represents absence of any influence, "
           "dashed red line represents a 50% increased spiking rate. "
    "(B) Relationship between baseline (120 ms before ripple start) and lateral ripple (0-50 ms) firing rate for clusters recorded in  HPF, Isocortex, MB and TH. In the Isocortex and MB plot "
    "we excluded the minority of clusters showing modulation >50% in response to either lateral or medial ripples (grey dots). Dashed black line represents absence of any influence, "
           "dashed red line represents a 50% increased spiking rate. ",

"Figure 5 - Figure supplement 2. Ripple modulation density histograms.":
"(A) Left: Early (0-50 ms) ripple modulation of hippocampal clusters in response to lateral and medial ripples. Dashed black line represents absence of any influence, "
           "dashed red line represents a 50% increased spiking rate. Wilcoxon signed-rank test. Right: "
"Late (50-120 ms) ripple modulation of hippocampal clusters in response to lateral and medial ripples. Dashed black line represents absence of any influence, "
           "dashed red line represents a 50% increased spiking rate. Wilcoxon signed-rank test. "
"(B) Ripple modulation of cortical (left), MB (middle) and TH (right) clusters in response to lateral and medial ripples. Dashed black line represents absence of any influence, "
           "dashed red line represents a 50% increased spiking rate. Wilcoxon signed-rank test",

"Figure 5 - Figure supplement 3. Cortical clusters showing ripple engagement.":
    "In pink clusters showing medial ripples engagement (at least 25%), in purple clusters showing lateral ripples engagement (at least 25%) and in red clusters showing "
    "engagement (at least 25%) both in medial and lateral ripples.",

"Figure 5 - Figure supplement 4. Ripple modulation across HPF, Isocortex, MB and TH.":
"(A) Ripple modulation in response to lateral and medial ripples during the early  "
           "ripple phase in cortical (top), MB (middle) and TH (bottom) clusters. Wilcoxon signed-rank test or Student's t-test (if normality established). "
"(B) Ripple modulation in response to lateral and medial ripples during the late  "
           "ripple phase in cortical (top), MB (middle) and TH (bottom) clusters. Wilcoxon signed-rank test or Student's t-test (if normality established). ",

"Figure 5 - Figure supplement 5. Pre-ripple modulation across HPF, Isocortex, MB and TH.":
"(A) Pre-ripple modulation in response to lateral and medial ripples during the early  "
           "ripple phase in cortical clusters. Wilcoxon signed-rank test or Student's t-test (if normality established). "
"(B) Ripple modulation in response to lateral and medial ripples during the late  "
           "ripple phase in MB clusters. Wilcoxon signed-rank test or Student's t-test (if normality established). " \
"(C) Ripple modulation in response to lateral and medial ripples during the late  "
           "ripple phase in TH clusters. Wilcoxon signed-rank test or Student's t-test (if normality established). ",

"Figure 5 - Figure supplement 6. Clusters preference in ripple engagement by hippocampal subfields.":
    "Preference in ripple engagement in CA1, CA3, DG, ProS and SUB."
}

if Adapt_for_Nature_style is True:
    legends_supplementary = Naturize_text(legends_supplementary)