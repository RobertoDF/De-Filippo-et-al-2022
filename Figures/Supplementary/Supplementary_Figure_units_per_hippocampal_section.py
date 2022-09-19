from Utils.Settings import output_folder_calculations, output_folder_supplementary, var_thr, waveform_PT_ratio_thr, isi_violations_thr, amplitude_cutoff_thr, presence_ratio_thr, Adapt_for_Nature_style
from Utils.Utils import Naturize
import dill
import pingouin as pg
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from Utils.Style import palette_ML
import pylustrator
import Utils.Style

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

#pylustrator.start()

fig, ax = plt.subplots(6, 2, figsize=(5,8))
param="Waveform duration"
ax0 = ax[0,0]
ax1 = ax[0,1]
legend = True
plot_distributions(total_units, ax0, ax1, param, legend)
param="Firing rate"
ax0 = ax[1,0]
ax1 = ax[1,1]
legend = False
plot_distributions(total_clusters[total_clusters["presence_ratio"]>presence_ratio_thr], ax0, ax1, param, legend)
ax[1,0].set_xlim(-5,20)
ax[1,1].set_xlim(-5,20)
param="Waveform amplitude"
ax0 = ax[2,0]
ax1 = ax[2,1]
plot_distributions(total_units, ax0, ax1, param, legend)
param="Waveform repolarization slope"
ax0 = ax[3,0]
ax1 = ax[3,1]
plot_distributions(total_units, ax0, ax1, param, legend)
param="Waveform recovery slope"
ax0 = ax[4,0]
ax1 = ax[4,1]
plot_distributions(total_units, ax0, ax1, param, legend)
ax[4,0].set_xlim(-.5,.1)
ax[4,1].set_xlim(-.5,.1)
param="Waveform PT ratio"
ax0 = ax[5,0]
ax1 = ax[5,1]
plot_distributions(total_units, ax0, ax1, param, legend)
ax[5,0].set_xlim(0,1)
ax[5,1].set_xlim(0,1)

fig.tight_layout()

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.036000, 0.981250])
plt.figure(1).texts[0].set_text("A")
plt.figure(1).texts[0].set_weight("bold")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.512000, 0.981250])
plt.figure(1).texts[1].set_text("B")
plt.figure(1).texts[1].set_weight("bold")

if Adapt_for_Nature_style is True:
    Naturize()

#% end: automatic generated code from pylustrator

#plt.show()
plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_11", dpi=300)



