from Utils.Settings import waveform_dur_thr, output_folder_figures_calculations, output_folder_calculations
import pandas as pd
import dill
import pingouin as pg
from scipy.stats import ks_2samp

# variables supplementary figure 1
with open(f"{output_folder_figures_calculations}/temp_data_supp_figure_1.pkl", 'rb') as f:
    data_area, ripple_power, session_id, session_summary, summary_table, ripples = dill.load(f)

session_id_supp_fig1 = session_id


with open(f'{output_folder_calculations}/ripples_features_sessions_across_all_ML.pkl', 'rb') as f:
    out = dill.load(f)

ripples_features_summary_strong = pd.concat([q[1] for q in out]).drop_duplicates()
ripples_features_summary_strong = ripples_features_summary_strong[ripples_features_summary_strong["Reference"]!="Central"]

ripples_features_summary_strong = ripples_features_summary_strong.infer_objects()

pg.ttest(ripples_features_summary_strong[ripples_features_summary_strong["Reference"]=="Medial"]["Strength conservation index"], ripples_features_summary_strong[ripples_features_summary_strong["Reference"]=="Lateral"]["Strength conservation index"])
out_test = pd.DataFrame(ripples_features_summary_strong\
            .pairwise_tukey(effsize="cohen", dv="Strength conservation index", between="Reference")[["A", "B", "p-tukey"]])
out_test = out_test[out_test["p-tukey"]<0.05]
pairs = list(out_test[["A", "B"]].apply(tuple, axis=1))
pvalues_supp_4 = out_test["p-tukey"].values[0]

with open(f'{output_folder_calculations}/trajectories_by_seed.pkl', 'rb') as f:
    trajectories_by_strength = dill.load(f)


with open(f"{output_folder_figures_calculations}/temp_data_figure_4.pkl", 'rb') as f:
    space_sub_spike_times, target_area, units, field_to_use_to_compare, \
    session_id_example, lfp, lfp_per_probe, \
    ripple_cluster_lateral_seed, ripple_cluster_medial_seed, source_area, ripples, \
    tot_summary_early, summary_fraction_active_clusters_per_ripples_early, \
    summary_fraction_active_clusters_per_ripples_early_by_neuron_type, \
    tot_summary_late, summary_fraction_active_clusters_per_ripples_late, \
    summary_fraction_active_clusters_per_ripples_late_by_neuron_type, \
    tot_summary, summary_fraction_active_clusters_per_ripples, \
    summary_fraction_active_clusters_per_ripples_by_neuron_type = dill.load(f)


data_sup_9_fraction_clu = summary_fraction_active_clusters_per_ripples.groupby(["Session id", "Location seed"]).mean().reset_index()

data_sup_9_spiking_rate = pd.melt(tot_summary[["Lateral seed", "Medial seed", "Session id"]],
               value_vars=["Lateral seed", "Medial seed"], id_vars=["Session id"], var_name='Location seed',
               value_name='Spiking rate per 10 ms').groupby(["Session id", "Location seed"]).mean().reset_index()

from Figures.Supplementary.Supplementary_Figure_spiking_rate_fraction_active_by_seed import ttest_clus_per_ripple, ttest_late_spiking


with open(f'{output_folder_calculations}/clusters_features_per_section.pkl', 'rb') as f:
    total_clusters = dill.load(f)

total_units = total_clusters[(total_clusters["waveform_PT_ratio"]<5)&(total_clusters["isi_violations"]<.5)&(total_clusters["amplitude_cutoff"]<.1)&(total_clusters["presence_ratio"]>.1)]



total_units_inh = total_units[total_units["Waveform duration"]<waveform_dur_thr]

param="waveform_duration"
p_val_wav_dur_inh = pg.mwu(total_units_inh[total_units_inh["Location"] == "Medial"][param],
       total_units_inh[total_units_inh["Location"] == "Lateral"][param])["p-val"][0]

param="waveform_recovery_slope"
p_val_wav_rec_slope_inh = pg.mwu(total_units_inh[total_units_inh["Location"] == "Medial"][param],
       total_units_inh[total_units_inh["Location"] == "Lateral"][param])["p-val"][0]

param="waveform_PT_ratio"
p_val_wav_PT_ratio_inh = pg.mwu(total_units_inh[total_units_inh["Location"] == "Medial"][param],
       total_units_inh[total_units_inh["Location"] == "Lateral"][param])["p-val"][0]


param="waveform_repolarization_slope"
p_val_wav_repolarization_slope_inh = pg.mwu(total_units_inh[total_units_inh["Location"] == "Medial"][param],
       total_units_inh[total_units_inh["Location"] == "Lateral"][param])["p-val"][0]


param="waveform_amplitude"
p_val_wav_amp_inh = pg.mwu(total_units_inh[total_units_inh["Location"] == "Medial"][param],
       total_units_inh[total_units_inh["Location"] == "Lateral"][param])["p-val"][0]


param="firing_rate"
p_val_fir_rate_inh = pg.mwu(total_units_inh[total_units_inh["Location"] == "Medial"][param],
       total_units_inh[total_units_inh["Location"] == "Lateral"][param])["p-val"][0]



param="waveform_duration"
p_val_ks_wav_dur_inh = ks_2samp(total_units_inh[total_units_inh["Location"] == "Medial"][param],
             total_units_inh[total_units_inh["Location"] == "Lateral"][param])[1]

param="waveform_recovery_slope"
p_val_ks_wav_rec_slope_inh = ks_2samp(total_units_inh[total_units_inh["Location"] == "Medial"][param],
             total_units_inh[total_units_inh["Location"] == "Lateral"][param])[1]


param="waveform_PT_ratio"
p_val_ks_wav_PT_ratio_inh = ks_2samp(total_units_inh[total_units_inh["Location"] == "Medial"][param],
             total_units_inh[total_units_inh["Location"] == "Lateral"][param])[1]


param="waveform_repolarization_slope"
p_val_ks_wav_repolarization_slope_inh = ks_2samp(total_units_inh[total_units_inh["Location"] == "Medial"][param],
             total_units_inh[total_units_inh["Location"] == "Lateral"][param])[1]

param="waveform_amplitude"
p_val_ks_wav_amp_inh = ks_2samp(total_units_inh[total_units_inh["Location"] == "Medial"][param],
             total_units_inh[total_units_inh["Location"] == "Lateral"][param])[1]


param="firing_rate"
p_val_ks_fir_rate_inh = ks_2samp(total_units_inh[total_units_inh["Location"] == "Medial"][param],
             total_units_inh[total_units_inh["Location"] == "Lateral"][param])[1]


total_units_exc = total_units[total_units["Waveform duration"]>=waveform_dur_thr]

param="waveform_duration"
p_val_wav_dur_exc = pg.mwu(total_units_exc[total_units_exc["Location"] == "Medial"][param],
       total_units_exc[total_units_exc["Location"] == "Lateral"][param])["p-val"][0]

param="waveform_recovery_slope"
p_val_wav_rec_slope_exc = pg.mwu(total_units_exc[total_units_exc["Location"] == "Medial"][param],
       total_units_exc[total_units_exc["Location"] == "Lateral"][param])["p-val"][0]

param="waveform_PT_ratio"
p_val_wav_PT_ratio_exc = pg.mwu(total_units_exc[total_units_exc["Location"] == "Medial"][param],
       total_units_exc[total_units_exc["Location"] == "Lateral"][param])["p-val"][0]


param="waveform_repolarization_slope"
p_val_wav_repolarization_slope_exc = pg.mwu(total_units_exc[total_units_exc["Location"] == "Medial"][param],
       total_units_exc[total_units_exc["Location"] == "Lateral"][param])["p-val"][0]


param="waveform_amplitude"
p_val_wav_amp_exc = pg.mwu(total_units_exc[total_units_exc["Location"] == "Medial"][param],
       total_units_exc[total_units_exc["Location"] == "Lateral"][param])["p-val"][0]


param="firing_rate"
p_val_fir_rate_exc = pg.mwu(total_units_exc[total_units_exc["Location"] == "Medial"][param],
       total_units_exc[total_units_exc["Location"] == "Lateral"][param])["p-val"][0]



param="waveform_duration"
p_val_ks_wav_dur_exc = ks_2samp(total_units_exc[total_units_exc["Location"] == "Medial"][param],
             total_units_exc[total_units_exc["Location"] == "Lateral"][param])[1]

param="waveform_recovery_slope"
p_val_ks_wav_rec_slope_exc = ks_2samp(total_units_exc[total_units_exc["Location"] == "Medial"][param],
             total_units_exc[total_units_exc["Location"] == "Lateral"][param])[1]


param="waveform_PT_ratio"
p_val_ks_wav_PT_ratio_exc = ks_2samp(total_units_exc[total_units_exc["Location"] == "Medial"][param],
             total_units_exc[total_units_exc["Location"] == "Lateral"][param])[1]


param="waveform_repolarization_slope"
p_val_ks_wav_repolarization_slope_exc = ks_2samp(total_units_exc[total_units_exc["Location"] == "Medial"][param],
             total_units_exc[total_units_exc["Location"] == "Lateral"][param])[1]

param="waveform_amplitude"
p_val_ks_wav_amp_exc = ks_2samp(total_units_exc[total_units_exc["Location"] == "Medial"][param],
             total_units_exc[total_units_exc["Location"] == "Lateral"][param])[1]


param="firing_rate"
p_val_ks_fir_rate_exc = ks_2samp(total_units_exc[total_units_exc["Location"] == "Medial"][param],
             total_units_exc[total_units_exc["Location"] == "Lateral"][param])[1]

