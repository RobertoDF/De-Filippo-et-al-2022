from Utils.Legends_variables import *
from Utils.Settings import Adapt_for_Nature_style
from Utils.Utils import Naturize_text

legends = {"Figure 1. Ripple strength correlation depends significantly on distance.":
                   f"(A) Correlation matrices showing the variabilty of ripple strength correlation between pairs of recording sites "
                   f"located in different CA1 locations in 4 example sessions. The number on the x and y axis labels indicates the probe number. Probes are numbered "
                   f"according to the position on the hippocampal longitudinal axis (0 is the most medial probe). "
                   f"(B) Scatter plot and linear regression showing the relationship between distance and correlation strength. "
                   f"Distance between recording sites explains {r_corr_distance_power_CA1_CA1}% of the variability in correlation of ripple strength. "
                   f"(C) Ripple strength correlation distribution. Pink represents bottom 25% (< Q₁) and blue top 25% (> Q₄). "
                   f"(D) Violinplots showing that the top and bottom correlation quartile show significantly different distance distributions ("
                   f"Q₁: {fig1_means_violinplot[0][0]} ± {fig1_sem_violinplot[0][0]} µm, Q₄: {fig1_means_violinplot[1][0]} ± {fig1_sem_violinplot[1][0]} µm, "
                   f"p-value = {'{:.2e}'.format(fig1_p_value_violin)}, Mann-Whitney U test). "
                   f"(E) Top: Rendering of the long distance (top) and short distance (bottom) CA1 pairs, "
                   f"dark circles are the reference locations in each pair. "
                   f"(F) Top and middle: scatter plots showing the relationship between ripple strength (at the reference location) and lag "
                   f"for long distance (top, n ripples = {fig1_n_high_dist}) and short distance (middle, "
                   f"n ripples = {fig1_n_low_dist}) pairs. Bottom: kernel density estimate of the lags of long distance (pink) "
                   f"and short distance (turquoise) pairs. "
                   f"(G): Lag (top) and absolute lag (bottom) comparison between long and short distance pairs (top: long distance ={fig1_mean_lagplot_summary[0]} ± {fig1_sem_lagplot_summary[0]} ms, "
                   f"Short distance = {fig1_mean_lagplot_summary[1]} ± {fig1_sem_lagplot_summary[1]} ms, p-value = { p_value_ttest_lag}, "
                   f"Student's t-test; "
                   f"bottom: long distance = {fig1_mean_abs_lagplot_summary[0]} ± {fig1_sem_abs_lagplot_summary[0]} ms, "
                   f"Short distance = {fig1_mean_abs_lagplot_summary[1]} ± {fig1_sem_abs_lagplot_summary[1]} ms, p-value = { p_value_ttest_abs_lag}, Student's t-test). "
                   f"(H) Lag comparison in long distance pairs between common and strong ripples with reference located in"
                   f"the medial (top) or lateral hippocampal section (bottom)"
                   f" (top: strong ripples={fig1_mean_lag_strong_ref_medial} ± {fig1_sem_lag_strong_ref_medial} ms, "
                   f"common ripples = {fig1_mean_lag_common_ref_medial} ± {fig1_sem_lag_common_ref_medial} ms, "
                   f"p-values = {p_value_ttest_lag_ref_medial}, Student's t-test, "
                   f"bottom: strong ripples={fig1_mean_lag_strong_ref_lateral} ± {fig1_sem_lag_strong_ref_lateral} ms, "
                   f"common ripples = {fig1_mean_lag_common_ref_lateral} ± {fig1_sem_lag_common_ref_lateral} ms, "
                   f"p-values = {p_value_ttest_lag_ref_lateral}, Student's t-test). "
                   f"(I) Lag comparison in long distance pairs between ripples with reference located in "
                   f"the medial and lateral section in common (top) or strong ripples (bottom)"
                   f" (top: medial reference = {fig1_mean_lag_ref_medial_common} ± {fig1_sem_lag_ref_medial_common} ms, "
                   f"lateral reference = {fig1_mean_lag_ref_lateral_common} ± {fig1_sem_lag_ref_lateral_common} ms, p-values = {p_value_ttest_common}, Student's t-test, "
                   f"bottom: strong ripples = {fig1_mean_lag_ref_medial_strong} ± {fig1_sem_lag_ref_medial_strong} ms, "
                   f"common ripples = {fig1_mean_lag_ref_lateral_strong} ± {fig1_sem_lag_ref_lateral_strong} ms, "
                   f"p-values = {p_value_ttest_strong}, Student's t-test).",
           
           "Figure 2. Direction-dependent differences in ripple propagation along the hippocampal longitudinal axis.":
                    f"(A) Recording locations for session {session_id_fig2}. Circles colors represents medio-lateral" \
                    f" location. Bigger circle represents the reference location. " \
                    f"(B) Example propagation of a strong (left column) and common (right column) ripple across the different recording location from session {session_id_fig2}, "
                    f"each filtered ripple is color-coded according to A. " \
                    f"Grey traces represents raw LFP signal. Dashed vertical line represents the start of the ripple. In the top row the ripple envelope across all locations. " \
                    f"Black scale bars: 50 ms, 0.5 mV. Red scale bars: 0.1 mV. "
                    f"(C) Average propagation map of strong and common ripples in session {session_id_fig2} across the medio-lateral axis. "
                    f"(D) Recording locations relative to E. Red circles represents the reference locations across all sessions (n sessions={n_sessions_fig2}), "
                    f"black circles represents the remaining recording locations. "
                    f"(E) Left: Medio-lateral propagation of strong ripples, each line represents the average of one session. "
                    f"Middle: Medio-lateral propagation of common ripples, each line represents the average of one session. "
                    f"Right: Average propagation map across sessions of strong and common ripples. Reference locations are the most lateral per session. "
                    f"(F) Same as D. "
                    f"(G) Same as E. Reference locations are the most lateral per session. "
                    f"(H) Same as D. "
                    f"(I) Same as E. Reference locations are the most central per session.",
                    
           "Figure 3. Ripples generation differences along the hippocampal longitudinal axis.":
           f"(A) Ripple seed location comparison between the three reference locations in "
           f"common ripples (left) and strong ripples (right). Majority of common ripples seeds are located in the lateral hippocampal "
           f"section regardless of the reference location (medial reference/lateral seed = {round(fig_3_mean_common.loc['Lateral seed','Medial'],2)} ± {round(fig_3_sem_common.loc['Lateral seed','Medial'],2)} %,"
           f" central reference/lateral seed = {round(fig_3_mean_common.loc['Lateral seed','Central'],2)} ± {round(fig_3_sem_common.loc['Lateral seed','Central'],2)} %,"
           f" lateral reference/lateral seed = {round(fig_3_mean_common.loc['Lateral seed','Lateral'],2)} ± {round(fig_3_sem_common.loc['Lateral seed','Lateral'],2)} %). "
           f"Strong ripples are mainly local (medial reference/medial seed = {round(fig_3_mean_strong.loc['Medial seed','Medial'],2)} ± {round(fig_3_sem_strong.loc['Medial seed','Medial'],2)} %,"
           f" central reference/central seed = {round(fig_3_mean_strong.loc['Central seed','Central'],2)} ± {round(fig_3_sem_strong.loc['Central seed','Central'],2)} %,"
           f" lateral reference/lateral seed = {round(fig_3_mean_strong.loc['Lateral seed','Lateral'],2)} ± {round(fig_3_sem_strong.loc['Lateral seed','Lateral'],2)} %)."
           f"(B) Ripple seed location comparison between strong and common ripples using a medial (left), "
           f"central (center) or lateral reference (right). "
           f"Asterisks mean p < 0.05, Kruskal-Wallis test with pairwise Mann-Whitney post-hoc test.",

            "Figure 4. Ripples travelling in the medio→lateral direction show prolonged network engagement.":
            f"(A) Recording location for session {session_id_fig4}. Circles colors indicate medio-lateral" \
                    f" location. Bigger circle represents the reference location. "
            f"(B) Spiking activity across the hippocampal M-L axis associated with a ripple generated medially (left column) "
            f"or lateraly (right column) across the different recording location from session {session_id_fig4}. Spike raster plot and normalized density"
            f" are plotted at each M-L location. "
            f"In the top row filtered ripple, grey traces represents raw LFP signal. All plots are  color coded according to A. " \
                    f"Scale bar: 0.5 mV. "
            f"(C) Kernel density estimates of the average spiking activity across different M-L locations and between seed type. "
            f"Scale bar: 5 spikes per 10 ms. "
            f"(D) Interpolated heatmap of the difference between medially and laterally generated ripple induced spiking activity in session {session_id_fig4}. Vertical dashed lines represent borders between early and late post-ripple start phases. Horizontal dashed lines represent the spatial limits of the hippocampal sections. "
            f"(E) Grand average of the differences between medially and laterally initiated ripple induced spiking activity "
            f"across {len([q[0] for q in list(spike_hists.keys()) if q[1] == 'HPF'])} sessions. Vertical dashed lines represent borders between early and late post-ripple start phases. "
            f"Horizontal dashed lines represent the spatial limits of the hippocampal sections. "
            f"(F) Regression plot between M-L location and ripple duration in common and strong ripples. "
            f"Horizontal dashed lines represent the spatial limits of the hippocampal sections. "
            f"(G) Average fraction of active neurons in medial (pink) and lateral (purple) ripples. Early/medial seed = {round(fig_4_summary_fraction_active_clusters_per_ripples_early[fig_4_summary_fraction_active_clusters_per_ripples_early['Location seed']=='Medial seed']['Fraction active neurons per ripple (%)'].mean(),2)}"
            f" ± {round(fig_4_summary_fraction_active_clusters_per_ripples_early[fig_4_summary_fraction_active_clusters_per_ripples_early['Location seed']=='Medial seed']['Fraction active neurons per ripple (%)'].sem() * 100,2)},"
            f" early/lateral seed: {round(fig_4_summary_fraction_active_clusters_per_ripples_early[fig_4_summary_fraction_active_clusters_per_ripples_early['Location seed']=='Lateral seed']['Fraction active neurons per ripple (%)'].mean()* 100,2)}"
            f" ± {round(fig_4_summary_fraction_active_clusters_per_ripples_early[fig_4_summary_fraction_active_clusters_per_ripples_early['Location seed']=='Lateral seed']['Fraction active neurons per ripple (%)'].sem()* 100,2)}, "
            f"p-value = {'{:.2e}'.format(fig_4_ttest_early_fraction['p-val'].values[0])}, Student's t-test; "
            f"late/medial seed = {round(fig_4_summary_fraction_active_clusters_per_ripples_late[fig_4_summary_fraction_active_clusters_per_ripples_late['Location seed']=='Medial seed']['Fraction active neurons per ripple (%)'].mean()* 100,2)}"
            f" ± {round(fig_4_summary_fraction_active_clusters_per_ripples_late[fig_4_summary_fraction_active_clusters_per_ripples_late['Location seed']=='Medial seed']['Fraction active neurons per ripple (%)'].sem()* 100,2)}, "
            f" late/lateral seed = {round(fig_4_summary_fraction_active_clusters_per_ripples_late[fig_4_summary_fraction_active_clusters_per_ripples_late['Location seed']=='Lateral seed']['Fraction active neurons per ripple (%)'].mean()* 100,2)}"
            f" ± {round(fig_4_summary_fraction_active_clusters_per_ripples_late[fig_4_summary_fraction_active_clusters_per_ripples_late['Location seed']=='Lateral seed']['Fraction active neurons per ripple (%)'].sem()* 100,2)}, "
            f"p-value = {'{:.2e}'.format(fig_4_ttest_late_fraction['p-val'].values[0])}, Student's t-test. "
            f"(H) Average spiking rate medial (pink) and lateral (purple) ripples. "
            f"Early/medial seed = {round(fig_4_summary_spiking_early[fig_4_summary_spiking_early['Location seed'] == 'Medial seed']['Spiking rate per 10 ms'].mean(), 2)} ± "
            f"{round(fig_4_summary_spiking_early[fig_4_summary_spiking_early['Location seed'] == 'Medial seed']['Spiking rate per 10 ms'].sem(), 3)}, "
            f"early/lateral seed = {round(fig_4_summary_spiking_early[fig_4_summary_spiking_early['Location seed'] == 'Lateral seed']['Spiking rate per 10 ms'].mean(), 2)} ± "
            f"{round(fig_4_summary_spiking_early[fig_4_summary_spiking_early['Location seed'] == 'Lateral seed']['Spiking rate per 10 ms'].sem(), 3)}, "
            f"p-value = {'{:.2e}'.format(fig_4_ttest_early_spiking['p-val'].values[0])}, Student's t-test; "
            f"late/medial seed ={round(fig_4_summary_spiking_late[fig_4_summary_spiking_late['Location seed'] == 'Medial seed']['Spiking rate per 10 ms'].mean(), 2)} ± "
            f"{round(fig_4_summary_spiking_late[fig_4_summary_spiking_late['Location seed'] == 'Medial seed']['Spiking rate per 10 ms'].sem(), 3)}, "
            f"late/lateral seed = {round(fig_4_summary_spiking_late[fig_4_summary_spiking_late['Location seed'] == 'Lateral seed']['Spiking rate per 10 ms'].mean(), 2)} ± "
            f"{round(fig_4_summary_spiking_late[fig_4_summary_spiking_late['Location seed'] == 'Lateral seed']['Spiking rate per 10 ms'].sem(), 3)}, "
            f"p-value = {'{:.2e}'.format(fig_4_ttest_late_spiking['p-val'].values[0])}, Student's t-test.",

           "Figure 5. Ripple seed location influences the pattern of ripple modulation across various regions of the brain.":
           "(A) Relationship between baseline (120 ms before ripple start) and ripple (0-120 ms) firing rate for clusters recorded in Isocortex, HPF, TH and MB. Dashed black line represents absence of any influence, "
           "dashed red line represents a 50% increased spiking rate. "
           "(B) Ripple modulation of hippocampal clusters in response to lateral and medial ripples. Dashed black line represents absence of any influence, "
           "dashed red line represents a 50% increased spiking rate. CLES=commn-language effect size. Wilcoxon signed-rank test. "
           "(C) Top: Rendering of all clusters recorded in the hippocampal formation color-coded by subfield. Middle: kernel density plot showing "
           "distribution of clusters along the M-L axis, Dashed lines represents medial and lateral limits. Bottom: Stacked kernel density plot showing "
           "distribution of clusters along the M-L axis. "
           "(D) Ripple modulation in response to lateral and medial ripples during the early (left) and late (right) "
           "ripple phase. Wilcoxon signed-rank test or Student's t-test (if normality established). "
           "(E) Ripple modulation in response to lateral and medial ripples before ripple start (20 ms). Wilcoxon signed-rank test or Student's t-test (if normality established). "
           "(F) Left: Relationship between modulation by lateral and medial ripples. Dashed black line represents absence of difference and two-fold differences in both directions. "
           "Right: Pie chart representing hippocampal neurons preference in ripple engagement."
           }


if Adapt_for_Nature_style is True:
    legends = Naturize_text(legends)




