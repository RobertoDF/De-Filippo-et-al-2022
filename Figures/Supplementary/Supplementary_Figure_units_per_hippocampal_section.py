from Utils.Settings import output_folder_calculations, output_folder_supplementary, var_thr, waveform_PT_ratio_thr, waveform_dur_thr , isi_violations_thr, amplitude_cutoff_thr, presence_ratio_thr, Adapt_for_Nature_style
from Utils.Utils import Naturize
import dill
import pandas as pd
import numpy as np
import pingouin as pg
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from Utils.Style import palette_ML
import Utils.Style


with open(f'{output_folder_calculations}/HPF_waveforms.pkl', 'rb') as f:
    exc_lat, exc_med, inh_lat, inh_med, time = dill.load(f)


with open(f'{output_folder_calculations}/clusters_features_per_section.pkl', 'rb') as f:
    total_clusters = dill.load(f)

total_units = total_clusters[(total_clusters["waveform_PT_ratio"]<waveform_PT_ratio_thr)&
                             (total_clusters["isi_violations"]<isi_violations_thr)&
                             (total_clusters["amplitude_cutoff"]<amplitude_cutoff_thr)&
                             (total_clusters["presence_ratio"]>presence_ratio_thr)]

total_units = total_units.rename(columns={"waveform_duration": "Waveform duration", "firing_rate":"Firing rate", "waveform_amplitude":"Waveform amplitude",
                                          "waveform_repolarization_slope":"Waveform repolarization slope", "waveform_recovery_slope":"Waveform recovery slope",
                                          "waveform_PT_ratio":"Waveform PT ratio"} )

total_clusters = total_clusters.rename(columns={"waveform_duration": "Waveform duration", "firing_rate":"Firing rate", "waveform_amplitude":"Waveform amplitude",
                                          "waveform_repolarization_slope":"Waveform repolarization slope", "waveform_recovery_slope":"Waveform recovery slope",
                                          "waveform_PT_ratio":"Waveform PT ratio"} )
def plot_distributions(total_units, ax0, ax1, param, legend):
    print(param)
    sns.kdeplot(ax=ax0, palette=palette_ML, data=total_units.query("Location=='Medial'| Location=='Lateral'"), x=param, hue="Location", common_norm=False)
    sns.ecdfplot(ax=ax1,  palette=palette_ML, data=total_units.query("Location=='Medial'| Location=='Lateral'"), x=param, hue="Location")
    norm_test = pg.normality(data=total_units, dv=param, group="Location")
    if norm_test["normal"].all():
        p_val = pg.ttest(total_units[total_units["Location"] == "Medial"][param],
               total_units[total_units["Location"] == "Lateral"][param])["p-val"][0]
        print("ttest: ", p_val)

    else:
        p_val = pg.mwu(total_units[total_units["Location"]=="Medial"][param], total_units[total_units["Location"]=="Lateral"][param])["p-val"][0]
        cles = pg.mwu(total_units[total_units["Location"]=="Medial"][param], total_units[total_units["Location"]=="Lateral"][param])["CLES"][0]
        print("mwu p-val and CLES: ", p_val, cles)

    if p_val<.05:
        ax0.text(.6, .7, "*",
                    transform=ax0.transAxes,
                    fontsize=15, ha='center', va='center');

    p_val_ks = ks_2samp(total_units[total_units["Location"] == "Medial"][param],
             total_units[total_units["Location"] == "Lateral"][param])[1]

    print("ks:", p_val_ks)
    if p_val_ks<.05:
        ax1.text(.6, .7, "*",
                    transform=ax1.transAxes,
                    fontsize=15, ha='center', va='center');

    if legend==False:
        ax0.get_legend().remove()
        ax1.get_legend().remove()



fig, ax = plt.subplots(8, 4, figsize=(10,10))

ax1 = plt.subplot2grid((8, 4), (0, 0), colspan=2, rowspan=2)

ax2 = plt.subplot2grid((8, 4), (0, 2), colspan=2, rowspan=2)

pd.DataFrame(np.vstack(exc_lat), columns=time*1000).mean().plot( color=palette_ML["Lateral"], ax=ax2)
pd.DataFrame(np.vstack(exc_med), columns=time*1000).mean().plot( color=palette_ML["Medial"], ax=ax2)

pd.DataFrame(np.vstack(inh_lat), columns=time*1000).mean().plot( color=palette_ML["Lateral"], ax=ax1)
pd.DataFrame(np.vstack(inh_med), columns=time*1000).mean().plot( color=palette_ML["Medial"], ax=ax1)

ax1.set_xlabel("Time (ms)")
ax2.set_xlabel("Time (ms)")

ax1.set_xlabel("Time (ms)")
ax2.set_xlabel("Time (ms)")

ax1.set_title("Putative inhibitory")
ax2.set_title("Putative excitatory")

ax1.text(-0.1, 1.15, "A", transform=ax1.transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

ax2.text(-0.1, 1.15, "B", transform=ax2.transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

param="Waveform duration"
ax0 = ax[2,0]
ax1 = ax[2,1]
legend = True
plot_distributions(total_units[total_units["Waveform duration"]<waveform_dur_thr], ax0, ax1, param, legend)
param="Firing rate"
ax0 = ax[3,0]
ax1 = ax[3,1]
legend = False
plot_distributions(total_units[(total_units["Waveform duration"]<waveform_dur_thr)&(total_units["presence_ratio"]>presence_ratio_thr)], ax0, ax1, param, legend)
ax[3,0].set_xlim(-5,50)
ax[3,1].set_xlim(-5,50)
param="Waveform amplitude"
ax0 = ax[4,0]
ax1 = ax[4,1]
plot_distributions(total_units[total_units["Waveform duration"]<waveform_dur_thr], ax0, ax1, param, legend)
param="Waveform repolarization slope"
ax0 = ax[5,0]
ax1 = ax[5,1]
plot_distributions(total_units[total_units["Waveform duration"]<waveform_dur_thr], ax0, ax1, param, legend)
param="Waveform recovery slope"
ax0 = ax[6,0]
ax1 = ax[6,1]
plot_distributions(total_units[total_units["Waveform duration"]<waveform_dur_thr], ax0, ax1, param, legend)
ax[6,0].set_xlim(-1,.1)
ax[6,1].set_xlim(-1,.1)
param="Waveform PT ratio"
ax0 = ax[7,0]
ax1 = ax[7,1]
plot_distributions(total_units[total_units["Waveform duration"]<waveform_dur_thr], ax0, ax1, param, legend)
ax[7,0].set_xlim(0,3)
ax[7,1].set_xlim(0,3)

param="Waveform duration"
ax0 = ax[2,2]
ax1 = ax[2,3]
legend = True
plot_distributions(total_units[total_units["Waveform duration"]>waveform_dur_thr], ax0, ax1, param, legend)
param="Firing rate"
ax0 = ax[3,2]
ax1 = ax[3,3]
legend = False
plot_distributions(total_units[(total_units["Waveform duration"]>waveform_dur_thr)&(total_units["presence_ratio"]>presence_ratio_thr)], ax0, ax1, param, legend)
ax[3,2].set_xlim(-5,20)
ax[3,3].set_xlim(-5,20)
param="Waveform amplitude"
ax0 = ax[4,2]
ax1 = ax[4,3]
plot_distributions(total_units[total_units["Waveform duration"]>waveform_dur_thr], ax0, ax1, param, legend)
param="Waveform repolarization slope"
ax0 = ax[5,2]
ax1 = ax[5,3]
plot_distributions(total_units[total_units["Waveform duration"]>waveform_dur_thr], ax0, ax1, param, legend)
param="Waveform recovery slope"
ax0 = ax[6,2]
ax1 = ax[6,3]
plot_distributions(total_units[total_units["Waveform duration"]>waveform_dur_thr], ax0, ax1, param, legend)
ax[6,2].set_xlim(-.5,.1)
ax[6,3].set_xlim(-.5,.1)
param="Waveform PT ratio"
ax0 = ax[7,2]
ax1 = ax[7,3]
plot_distributions(total_units[total_units["Waveform duration"]>waveform_dur_thr], ax0, ax1, param, legend)
ax[7,2].set_xlim(0,1)
ax[7,3].set_xlim(0,1)
fig.tight_layout()


#plt.show()

plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_13", dpi=300)




