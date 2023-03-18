from Utils.Results_variables import *
from Utils.Settings import Adapt_for_Nature_style
from Utils.Utils import Naturize_text

results = {"Distance explains most of the ripple strength correlation variability.":
            f"We studied ripple propagation along the hippocampal longitudinal axis in" \
             " an open-access dataset provided by the Allen Institute. "
            "We analyzed the LFP signals across the visual cortex, hippocampal formation and brain stem (Supplementary Figure 1) simultaneous to ripples detected "
            f"in the CA1 of {summary_table['Session'].unique().shape[0]} animals (average session duration = "
            f"{round(np.mean(list(sessions_durations.values())),1)} ± {round(sem(list(sessions_durations.values())),1)} seconds, average ripple incidence during non-running epochs = "
            f"{round(np.mean(list(ripple_freq_total.values()))*10,2)} ± {round(sem(list(ripple_freq_total.values()))*10,2)} per 10s). "
            f"Ripples (n ripples = {np.sum(list( number_ripples_per_session_best_CA1_channel.values()))}) "
            f"were detected on the CA1 channel with the strongest ripple activity. "
            f"Ripple strength (∫Ripple) was calculated as the integral of the filtered LFP envelope "
            f"between the start and end points for every detected ripple. "
            f"Ripple strength and duration are highly correlated in each session (mean r = {round(r_ripple_duration_amplitude_list.mean()[0],2)} ± "
            f"{round(r_ripple_duration_amplitude_list.sem()[0],3)}, Supplementary Figure 2). "
            f"Notably ripple strength correlates significantly better "
            f"with the hippocampal population spiking rate on a ripple-to-ripple basis compared to ripple duration alone (p-value = {t_test_corr_spikes_vs_dur_or_strength}, Supplementary Figure 3). "
            f"Clear ripples were observed uniquely in "
            f"the hippocampal formation (CA1, CA2, CA3, DG, SUB, ProS). Likewise, ripple-induced voltage "
            f"deflections (RIVD, integral of the unfiltered LFP envelope) "
            f"were also noticeably stronger in hippocampal areas (Supplementary Figure 4B-F). Ripple strength was noticeably irregular in single sessions both across time and space, even"
            f" within the CA1 region (Supplementary Figure 4C). "
            f"We focused on the variability in ripple strength across pairs of CA1 recording locations "
            f"with clear ripple activity (n CA1 pairs = {summary_corrs[summary_corrs['Comparison'] == 'CA1-CA1'].shape[0]},"
            f" n sessions = {summary_corrs['Session'].unique().shape[0]}). "
            f"Correlation of ripple strength across different CA1 regions was highly variable (Figure 1A-B-C) "
            f"with a lower and upper quartiles of {round(quartiles_correlation[0.25],2)} and {round(quartiles_correlation[0.75],2)} "
            f"(mean = {round(summary_corrs[summary_corrs['Comparison'] == 'CA1-CA1']['Correlation'].mean(),2)}, "
            f"SEM = {round(summary_corrs[summary_corrs['Comparison'] == 'CA1-CA1']['Correlation'].sem(),2)}). "
            f"Distance between recording location could explain the majority ({round(r_squared_corr_distance*100, 2)} %) of this variability (Figure 1B) "
            f"with the top and bottom quartiles of ripple strength correlation showing significantly different average distances (Figure 1C-D). "
            f"Given the correlation variability we asked how reliably a ripple can travel along the hippocampal longitudinal axis. "
            f"To answer this question, we looked at ripples lag in sessions that included both long-distance (> {round(quartiles_distance[0.75], 2)} µm) and "
            f"short-distance (< {round(quartiles_distance[0.25], 2)} µm) CA1 recording pairs (n sessions = {ripples_lags['Session'].unique().shape[0]}, n CA1 pairs = {ripples_lags['Session'].unique().shape[0] * 2}, Figure 1E). "
            f"Reference for the lag analysis was always the most medial recording location in each pair. "
            f"Almost half of the ripples in long-distance pairs ({round((ripples_lags_clipped[ripples_lags_clipped['Type']=='High distance (µm)'].groupby('Session').size()/ripples_lags[ripples_lags['Type']=='High distance (µm)'].groupby('Session').size()).mean()*100, 2)} ± "
            f"{round((ripples_lags_clipped[ripples_lags_clipped['Type']=='High distance (µm)'].groupby('Session').size()/ripples_lags[ripples_lags['Type']=='High distance (µm)'].groupby('Session').size()).sem()*100, 2)} %) "
            f"were detected in both locations (inside a {np.sum(np.abs(clip_ripples_clusters))} ms window centered on ripple start at the reference location). "
            f"Unsurprisingly short-distance pairs showed a more reliable propagation ({round((ripples_lags_clipped[ripples_lags_clipped['Type']=='Low distance (µm)'].groupby('Session').size()/ripples_lags[ripples_lags['Type']=='Low distance (µm)'].groupby('Session').size()).mean()*100, 2)} ± "
            f"{round((ripples_lags_clipped[ripples_lags_clipped['Type']=='Low distance (µm)'].groupby('Session').size()/ripples_lags[ripples_lags['Type']=='Low distance (µm)'].groupby('Session').size()).sem()*100, 2)} %). "
            f"Moreover, lag between long-distance pairs had a much broader distribution (Figure 1F) and a "
            f"significantly bigger absolute lag (Figure 1G). Neither high nor short-distance pairs showed clear directionality "
            f"(lag long-distance = {round(ripples_lags_clipped[ripples_lags_clipped['Type']=='High distance (µm)'].groupby('Session')['Lag (ms)'].mean().mean(),2)} ± {round(ripples_lags_clipped[ripples_lags_clipped['Type']=='High distance (µm)'].groupby('Session')['Lag (ms)'].mean().sem(),2)} ms, "   
            f"lag short-distance = {round(ripples_lags_clipped[ripples_lags_clipped['Type']=='Low distance (µm)'].groupby('Session')['Lag (ms)'].mean().mean(),2)} ± {round(ripples_lags_clipped[ripples_lags_clipped['Type']=='Low distance (µm)'].groupby('Session')['Lag (ms)'].mean().sem(),2)} ms). "
            f"Looking at the relationship between lag and ripple strength in long-distance pairs, however, an asymmetric distribution was apparent (Figure 1F top), "
            f"suggestive of a possible interaction between these two variables: stronger ripples appear to be predominantly associated with positive lags (i.e. ripples moving medial→lateral). "
            f"To further investigate this relationship we divided ripples into two groups: strong (top 10% ripple strength per session at the reference location) "
            f"and common (remaining ripples). "
            f"The vast majority of the variance in 3D distance between recording locations was explained by the distance "
            f"on the medio-lateral (M-L) axis alone (R² = {round(corr_table_distance['Distance (µm)']['M-L (µm)']**2,3)}). "
            f"To simplify our analysis we therefore focused only on this spatial dimension. "
            f"The septal half of the hippocampus was therefore divided along the M-L axis in three sections with equal number of recordings:"
            f" medial, central and lateral (Supplementary Figure 5). "
            f"Strong ripples identified in the medial section, in opposition to common ripples, showed a markedly positive lag "
            f"(lag = {fig1_mean_lag_strong_ref_medial} ± {fig1_sem_lag_strong_ref_medial} ms) indicative of a "
            f"preferred medial→lateral travelling direction (Figure 1H top). "
            f"Surprisingly, the same was not true for strong ripples identified in the lateral section (lag = {fig1_mean_lag_strong_ref_lateral} ± {fig1_sem_lag_strong_ref_lateral} ms, Figure 1I). "
            f"Strong and common ripples lags were significantly different between medial and lateral locations both in common and strong ripples. "
            f"A biased direction of propagation can be explained by an unequal chance of ripple generation across space. "
            f"We can assume that selecting strong ripples we are biasing our focus towards ripples whose generation point (seed) is situated nearby our reference "
            f"location, this would contribute to explain the unbalanced lag. This notion would, however, fail to explain the different directionality we observed between strong ripples"
            f" in medial and lateral locations. This hints at a more complex situation.",

            "Ripples propagates differentially along the hippocampal longitudinal axis.":
            f"To analyze the propagation of ripples along the hippocampal longitudinal axis we focused on sessions from which ripples were clearly "
            f"detected in at least two different hippocampal sections at the same time (n = {trajectories_by_strength['Session'].unique().shape[0]}). "
            f"We followed the propagation of strong and common ripples detected in the reference location across the hippocampus (Figure 2A-B) and built an average spatio-temporal propagation map "
            f"per session (Figure 2C). "
            f"Strong and common ripples in the medial section showed a divergent propagation pattern: strong ripples travelling medio→laterally"
            f" and common ripples travelling in the opposite direction (Figure 2D-E). "
            f"Ripples detected in the lateral section did not show such strikingly divergent propagation (Figure 2F-G) whereas, in the central section,"
            f" the propagation was "
            f"divergent only laterally and not medially (Figure 2H-I). This peculiar propagation profile suggests "
            "a not previously described underlying directionality along the hippocampal longitudinal axis and can be possibly explained "
            "by a spatial bias in strong ripples generation. "
            f"To understand the mechanism underlying such difference in propagation "
            f"we examined the location of the seed for each ripple in sessions in which ripples were clearly detected "
            f"in every hippocampal section (n sessions = {seed_ripples_by_hip_section_summary_strong['Session id'].unique().shape[0]})"
            f". While we found no differences in the number of ripples detected in each hippocampal "
            f"section (p-value = {round(p_value_ripples_per_section[0], 2)}, Kruskal-Wallis test), "
            f"we observed differences regarding ripple generation."
            f" In common ripples, regardless of the reference location,"
            f" most ripples started from the lateral section (Figure 3A left)."
            f" On the other hand, strong ripples displayed a more heterogenous picture (Figure 3A right). We identified two principles relative to "
            f"strong ripples generation:"
            f" In all hippocampal sections the majority of strong ripples are locally generated, and a greater number of strong ripples is generated medially than laterally. "
            f"Looking at the central section we can appreciate the difference between the number of strong ripples "
            f"generated medially and laterally (Figure 3A right, mean medial = {round(seed_ripples_by_hip_section_summary_strong[seed_ripples_by_hip_section_summary_strong['Reference']=='Central'].groupby('Location seed').mean().loc['Medial seed']['Percentage seed (%)'], 2)}"
            f" ± {round(seed_ripples_by_hip_section_summary_strong[seed_ripples_by_hip_section_summary_strong['Reference']=='Central'].groupby('Location seed').sem().loc['Medial seed']['Percentage seed (%)'], 2)} %, "
            f"mean lateral = {round(seed_ripples_by_hip_section_summary_strong[seed_ripples_by_hip_section_summary_strong['Reference']=='Central'].groupby('Location seed').mean().loc['Lateral seed']['Percentage seed (%)'], 2)} ± "
            f"{round(seed_ripples_by_hip_section_summary_strong[seed_ripples_by_hip_section_summary_strong['Reference']=='Central'].groupby('Location seed').sem().loc['Lateral seed']['Percentage seed (%)'], 2)} %, "
            f"p-value = {p_val_medial_lateral}, Pairwise Tukey test). Strong and common ripples had significantly different seed location profiles only in the medial and central section, "
            f"not in the lateral section (Figure 3B). "
            f"These seed location profiles contribute to explain "
            f"the propagation idiosyncrasies: major unbalances in seeds location cause propagation patterns with clear directionality, "
            f"on the contrary, lag measurements hovering around zero are the result of "
            f"averaging between two similarly numbered groups of ripples with opposite direction of propagation. Notably, "
            f"propagation speed did not change depending on the seed location (Supplementary Figure 6) and the antero-posterior (A-P) axis "
            f"did not explain a considerable amount of variability in lag (R² medial = {round(res_medial_local_ap['r'][0]**2,4)} , "
            f"p-value = {'{:.2e}'.format(res_medial_local_ap['p-val'][0])}, "
            f"R² lateral = {round(res_lateral_local_ap['r'][0]**2,4)}, p-value = {'{:.2e}'.format(res_lateral_local_ap['p-val'][0])}),"
            f" in contrast with the M-L axis "
            f"(R² medial = {round(res_medial_local_ml['r'][0]**2,4)}, p-value = {'{:.2e}'.format(res_medial_local_ml['p-val'][0])}, "
            f"R² lateral = {round(res_lateral_local_ml['r'][0]**2,4)}, p-value = {'{:.2e}'.format(res_lateral_local_ml['p-val'][0])}). "
            f"The reason why strong ripples are only in a minority of cases"
            f" generated in the lateral section remains nevertheless unclear. Using a 'strength conservation index' (SCI) "
            f"we measured the ability of a ripple to retain its strength during propagation "
            f"(a ripple with SCI = 1 is in the top 10% in all hippocampal sections). We observed that ripples generated laterally "
            f"were effectively less able to retain their strength propagating towards the medial pole (Supplementary Figure 7). "
            f"This result is not simply "
            f"explained by differences in ripple strength along the M-L axis,"
            f" as no such gradient was observed (R² = {round(r_ML_strength**2,4)}, Supplementary Figure 8). Curiously, "
            f"ripple amplitude showed a weak trend in the opposite direction (r = {round(r_ML_amp,2)}, p-value = {'{:.2e}'.format(p_ML_amp)}), "
            f"with higher amplitude ripples in the lateral section (Supplementary Figure 9).",

            "The hippocampal medial pole can generate longer ripples able to better engage neural networks.":
            f"To understand the reason behind the differential propagation we focused uniquely on the central section, "
            f"here it was possible to distinguish between ripples generated laterally or medially ('lateral ripples' and 'medial ripples'). We "
            f"focused solely on ripples generated in the lateral and medial section, discarding the ones generated in the central section."
            f"We included in the analysis sessions in which ripples were clearly detected in each hippocampal section and with at least {minimum_ripples_count_generated_in_lateral_or_medial_spike_analysis}"
            f" ripples of each kind (n sessions = {len([q[0] for q in list(spike_hists.keys()) if q[1] == 'HPF'])}). "
            f"We looked at spiking activity associated with these "
            f"two classes of ripples "
            f"in the hippocampal formation across the M-L axis (n clusters per session = {round(tot_summary.groupby('Session id').size().mean(), 2)} ± {round(tot_summary.groupby('Session id').size().sem(), 2)}, Figure 4A-B-C). "
            f"To compare sessions, we created interpolated maps of the difference between spiking induced"
            f' by medial and lateral ripples (Figure 4D). Immediately following ripple start (0-50 ms, "early phase") spiking was '
            f"predictably influenced by ripple seed proximity: in the lateral section lateral ripples induced more spiking (indicated by the blue color), "
            f"whereas in the medial section medial ripples dominated (indicated by the red color). "
            f'Surprisingly, in the 50-120 ms window post ripple start ("late phase"), medial ripples could elicit significantly higher '
            f"spiking activity than lateral ripples along"
            f" the entire M-L axis (Figure 4E). Dividing clusters in putative excitatory and inhibitory using the waveform duration "
            f"we observed the same effect in both types of neurons (Supplementary Figure 10). "
            f"In accordance with this result, we found that the medial hippocampal section is able to generate longer "
            f"ripples (Figure 4F). An important portion of the variance "
            f"in ripple duration is indeed explained by location on the M-L axis both in common (R² = {r_common}) and especially "
            f"in strong ripples (R² = {r_strong}). "
            f"The observed extended spiking could be due to a increased number of neurons participating in the ripple,"
            f" to a higher spiking rate per neuron or a combination of "
            f"these two elements. "
            f"Fraction of active neurons and spiking rate were both significantly higher in medial ripples (Supplementary Figure 11). "
            f"Focusing only on the late phase "
            f"the difference in fraction of active neurons per ripples between medial and lateral ripples was even more "
            f"striking (Cohen's d = {str(round(ttest_late_clus_per_ripple['cohen-d'].values[0], 2))}, Figure 4G). Inversely, "
            f"in the early phase, lateral ripples could engage more neurons, although, the effect size was much smaller "
            f"(Cohen's d = {str(round(ttest_early_clus_per_ripple['cohen-d'].values[0], 2))}). The same result was found in relation to the spiking rate, medial ripples "
            f"caused a significant and considerable increase in spiking rate in the late phase (Cohen's d = {str(round(ttest_late_spiking['cohen-d'].values[0], 2))}, Figure 4H). "
            f"Dividing again the clusters into putative excitatory and inhibitory, significant differences between medial and lateral ripples "
            f"were present only in the late phase. Spiking frequency and number of engaged neurons were significantly higher in medial ripples both in putative excitatory"
            f" and inhibitory clusters (Supplementary Figure 12). In summary, the prolonged spiking observed in medial ripples was caused both by an increased"
            f" number of engaged neurons and a higher spiking rate per cell, both in putative excitatory and inhibitory neurons. "
            f"The disparity in network engagement can possibly "
            f"be in part explained by electrophysiological differences across hippocampal sections (e.g. higher firing rate). "
            f"We did not find differences in the number of firing neurons across the entire hippocampal formation (medial = {round(normalized_cluster_count_per_probe['Medial'].mean(), 2)}, "
            f"lateral = {round(normalized_cluster_count_per_probe['Lateral'].mean(), 2)}, p-value = {'{:.2e}'.format(test_cluster_count)}, Mann-Whitney U test), " 
            f"we did, however, found differences in firing rate, waveform duration, and waveform shape (recovery slope and peak-through ratio, Supplementary Figure 13)."
            f" Firing rate and waveform duration in putative excitatory neurons exhibited respectively "
            f"a left- and right-shifted distribution in the lateral section, reflecting lower firing rate and slower action potentials. "
            f"Putative inhibitory interneurons in the lateral section showed both a slower action potential and a higher firing rate.",

           "Location of ripple seed is associated with different pattern of modulation across brain regions":
           "To investigate how medial and lateral ripples affect various brain regions, we examined the modulation of spiking rate during ripples in individual clusters. "
            "A cluster was deemed modulated if it exhibited at least a 50% increase in spiking rate during ripples. "
           f'We found that clusters located in the thalamus (TH) and midbrain (MB) were hardly modulated (MB: {round(100-ripple_mod_mean.loc["MB"]["No engagement"]*100, 2)} ± {round(ripple_mod_sem.loc["MB"]["No engagement"] *100, 2)} %, '
           f'TH: {round(100-ripple_mod_mean.loc["TH"]["No engagement"]*100, 2)} ± {round(ripple_mod_sem.loc["TH"]["No engagement"] *100, 2)} %), ' \
           'with baseline spiking rate explaining nearly all the variance in spiking rate during ripples. In contrast, only a small fraction of '
           f'cortical neurons were modulated (Isocortex: {round(100-ripple_mod_mean.loc["Isocortex"]["No engagement"]*100, 2)} ± '
           f'{round(ripple_mod_sem.loc["Isocortex"]["No engagement"] *100, 2)} %))  while the majority of hippocampal (HPF) neurons showed ripple modulation '
           f'(HPF: {round(100-ripple_mod_mean.loc["HPF"]["No engagement"]*100, 2)} ± {round(ripple_mod_sem.loc["HPF"]["No engagement"] *100, 2)} %). '
           f'In both cases the baseline spiking rate explained a big portion of the variance in spiking rate during ripples (Figure 5A).'
           f'Neurons in the cortex'
           f' that were both modulated by medial and lateral ripples were generally found in deeper layers, '
           f'whereas neurons modulated more strongly by medial ripples were'
           f' also found in more superficial layers. Only a small number of cortical neurons exhibited greater engagement in association with lateral ripples (Supplementary Figure 14). '
           f' The relationship between'
           f' baseline spiking and spiking during ripples was similar in medial and lateral ripples (Supplementary Figure 15). '
           f'This similarity accounted for most of the variability observed in all cases. '
           f'There was a significant difference in modulation of hippocampal neurons by medial and lateral ripples within a 120 ms window after the start of the ripple event. '
           f'However, the effect size was modest (Figure 5B). A stronger effect was observed in the 50-120 ms window, here medial ripples had a notably stronger '
           f'modulatory effect (Supplementary Figure 16A). Significant differences were also observed in cortical neurons and in TH, with medial ripples '
           f'inducing stronger modulation (Supplementary Figure 16B). To understand the mechanism underlying these differences between medial and lateral ripples '
           f'we focused on modulation in the early (0-50 ms) and late phase (50-120 ms) in various hippocampal subfield (Figure 5C). In the early phase '
           f'we found a significantly stronger engagement of the dentate gyrus (DG), CA1 and CA3 areas and weaker engagement of the subiculum (SUB) by lateral ripples. '
           f'The late phase analysis revealed further differences, as CA1, CA3, DG, prosubiculum (ProS), and SUB subfields all displayed stronger modulation in response to medial ripples '
           f'(Figure 5D). '
           f'The clusters recorded across these various brain regions were not uniformly distributed along the M-L axis, which may have influenced the observed ripple modulation. '
           f'For instance, it is conceivable that neurons located in closer proximity to the medial section may be more susceptible to the effects of medial ripples. '
           f'To examine this hypothesis, we evaluated the degree to which variance in modulation could be attributed to the M-L axis position.'
           f'We observed that during the early and late phases, the M-L position explained a substantial proportion '
           f'of the variance in modulation in CA1 (early: {round(r_sq_ca1.loc["Diff ripple modulation (0-50 ms)", "M-L"]*100, 2)} %, late: '
           f'{round(r_sq_ca1.loc["Diff ripple modulation (50-120 ms)", "M-L"]*100, 2)} %), but only a small proportion in CA3 (early: {round(r_sq_ca3.loc["Diff ripple modulation (0-50 ms)", "M-L"]*100, 2)} %, '
           f'late: {round(r_sq_ca3.loc["Diff ripple modulation (50-120 ms)", "M-L"]*100, 2)} %) ProS '
           f'(late: {round(r_sq_pros.loc["Diff ripple modulation (50-120 ms)", "M-L"]*100, 2)} %) and SUB '
           f'(early: {round(r_sq_sub.loc["Diff ripple modulation (0-50 ms)", "M-L"]*100, 2)} %, '
           f'late: {round(r_sq_sub.loc["Diff ripple modulation (50-120 ms)", "M-L"]*100, 2)} %). In the DG subfield, the situation was mixed, with a notably greater impact '
           f'of M-L position observed during the late phase '
           f'(early: {round(r_sq_dg.loc["Diff ripple modulation (0-50 ms)", "M-L"]*100, 2)} %, '
           f'late: {round(r_sq_dg.loc["Diff ripple modulation (50-120 ms)", "M-L"]*100, 2)} %).'
           f'Additionally, '
           f'we found that prior to the onset of ripples, medial ripples induced stronger modulation in ProS and SUB, and weaker modulation in CA3 '
           f'and DG when compared to lateral ripples (Figure 5E). In this window, the M-L position explained only a minimal proportion of the variance '
           f'in ProS ({round(r_sq_pros.loc["Diff pre-ripple modulation (20-0 ms)", "M-L"]*100, 2)} %), SUB '
           f'({round(r_sq_sub.loc["Diff pre-ripple modulation (20-0 ms)", "M-L"]*100, 2)} %) and CA3 '
           f'({round(r_sq_ca3.loc["Diff pre-ripple modulation (20-0 ms)", "M-L"]*100, 2)} %) but a substantial proportion in DG '
           f'({round(r_sq_dg.loc["Diff pre-ripple modulation (20-0 ms)", "M-L"]*100, 2)} %). In both the early and late ripple phases (Supplementary Figure 17), '
           f'medial ripples elicited significantly stronger modulation across multiple cortical regions. '
           f'This effect was also observed before the onset of ripples (Supplementary Figure 18). '
           f'Furthermore, we investigated whether neurons exhibit a preference for medial or lateral ripples. '
           f'To address this question, we identified neurons that exhibited at least a two-fold modulation '
           f'difference for either medial or lateral ripples. Our analysis revealed that the majority of neurons did '
           f'not display a strong preference ({round(autopcts[0]*100,2)} %), whereas {round(autopcts[1]*100,2)} % of neurons responded '
           f'preferentially to lateral ripples and {round(autopcts[2]*100,2)} % to medial ripples (Figure 5F). Hippocampal subfield, however, exhibited '
           f'striking differences, specifically, SUB had a small proportion of neurons exhibiting a'
           f' preference for medial ripples, while DG exhibited the opposite pattern (Supplementary Figure 19). '
           f'These results suggest a fundamental difference between medial and lateral ripples in terms of their '
           f'engagement of brain regions, highlighting that these differences exist even before the ripples are detected. '
           f'Notably, most of these differences are not explained by the location of the neuron along the M-L axis. '
           f'This suggests that there are more fundamental differences at play beyond just spatial distance. While the M-L '
           f'position did explain some of the variance in modulation in certain subfields, such as CA1 and DG, it only accounted '
           f'for a small proportion of the variance in others, like CA3 and SUB. '


           }

if Adapt_for_Nature_style is True:
    results = Naturize_text(results)