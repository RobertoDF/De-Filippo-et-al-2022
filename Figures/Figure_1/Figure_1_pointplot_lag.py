import pingouin as pg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from Utils.Style import palette_ML, palette_timelags
from Utils.Settings import output_folder_figures_calculations, clip_ripples_clusters
import Utils.Style

with open(f"{output_folder_figures_calculations}/temp_data_figure_1.pkl", "rb") as fp:  # Unpickling
    sessions, high_distance, low_distance, ripples_lags, ripples_lags_inverted_reference, ripples_calcs, summary_corrs = pickle.load(fp)

ripples_lags = ripples_lags[ripples_lags["Lag (ms)"].between(clip_ripples_clusters[0], clip_ripples_clusters[1])]

quant = 0.9
ripples_lags_strong = ripples_lags.groupby(['Session', "Type"]).apply(lambda group: group[group["∫Ripple"] >= group["∫Ripple"].quantile(quant)]).reset_index(drop=True)
ripples_lags_strong["Quantile"] = "Strong ripples"

ripples_lags_weak = ripples_lags.groupby(['Session', "Type"]).apply(lambda group: group[group["∫Ripple"] < group["∫Ripple"].quantile(quant)]).reset_index(drop=True)
ripples_lags_weak["Quantile"] = "Common ripples"

ripples_lags_by_percentile = pd.concat([ripples_lags_strong, ripples_lags_weak])
ripples_lags_by_percentile["Reference"] = "Medial"

ripples_lags_inverted_reference = ripples_lags_inverted_reference[ripples_lags_inverted_reference["Lag (ms)"].between(clip_ripples_clusters[0], clip_ripples_clusters[1])]

quant = 0.9
ripples_lags_inverted_reference_strong = ripples_lags_inverted_reference.groupby(['Session', "Type"]).apply(lambda group: group[group["∫Ripple"] > group["∫Ripple"].quantile(quant)]).reset_index(drop=True)
ripples_lags_inverted_reference_strong["Quantile"] = "Strong ripples"

ripples_lags_inverted_reference_weak = ripples_lags_inverted_reference.groupby(['Session', "Type"]).apply(lambda group: group[group["∫Ripple"] < group["∫Ripple"].quantile(quant)]).reset_index(drop=True)
ripples_lags_inverted_reference_weak["Quantile"] = "Common ripples"

ripples_lags_inverted_reference_by_percentile = pd.concat([ripples_lags_inverted_reference_strong, ripples_lags_inverted_reference_weak])
ripples_lags_inverted_reference_by_percentile["Reference"] = "Lateral"

summary_lags = pd.concat([ripples_lags_by_percentile, ripples_lags_inverted_reference_by_percentile])

# reference comparison
data = summary_lags[summary_lags["Type"] == "High distance (µm)"]

fig, axs = plt.subplots(2, 2, figsize=(3, 4))
# sns.stripplot(x="Lag (ms)", y="Type", hue="Session",
#               data=tot_diffs[tot_diffs["Lag (ms)"].between(-4,4)], dodge=True, alpha=.25, zorder=1)
g = sns.pointplot(x="Lag (ms)", y="Reference", hue="Session",
                  data=data[data["Quantile"]=="Strong ripples"], dodge=0.3,
                  join=True,
                  markers="o", scale=0.8, ci=None, palette=sns.color_palette("Greys", 80)[20:], ax=axs[1, 0])
plt.setp(g.collections, alpha=.6)  # for the markers
plt.setp(g.lines, alpha=.6)
g.legend([], [], frameon=False)

gg = sns.pointplot(x="Lag (ms)", y="Reference",
                   data=data[data["Quantile"]=="Strong ripples"].groupby(["Session", "Reference"]).mean().reset_index(),
                   join=True,
                   markers="o", scale=1.5, ci=95, palette=palette_ML, ax=axs[1, 0], order=["Medial", "Lateral"])
plt.setp(gg.collections, zorder=100)  # for the markers
plt.setp(gg.lines, zorder=100)
axs[1, 0].set_title("Strong ripples", color=palette_timelags["Strong ripples"])

g = sns.pointplot(x="Lag (ms)", y="Reference", hue="Session",
                  data=data[data["Quantile"]=="Common ripples"], dodge=0.3,
                  join=True,
                  markers="o", scale=0.8, ci=None, palette=sns.color_palette("Greys", 80)[20:], ax=axs[0, 0])
plt.setp(g.collections, alpha=.6)  # for the markers
plt.setp(g.lines, alpha=.6)
g.legend([], [], frameon=False)

gg = sns.pointplot(x="Lag (ms)", y="Reference",
                   data=data[data["Quantile"]=="Common ripples"].groupby(["Session", "Reference"]).mean().reset_index(),
                   join=True,
                   markers="o", scale=1.5, ci=95, palette=palette_ML, ax=axs[0, 0], order=["Medial", "Lateral"])
plt.setp(gg.collections, zorder=100);  # for the markers
plt.setp(gg.lines, zorder=100)
axs[0, 0].set_ylabel('')
axs[0, 0].set_title("Common ripples", color=palette_timelags["Common ripples"])

axs[1, 0].set_ylabel('')


# print(pg.normality(data[data["Quantile"]=="Common ripples"].groupby(["Session", "Reference"]).mean().reset_index(), dv="Lag (ms)", group="Reference", method="normaltest"))
ttest = pg.ttest(data[(data["Quantile"]== "Common ripples") & (data["Reference"]== "Medial")].groupby("Session").mean()["Lag (ms)"],
                 data[(data["Quantile"]== "Common ripples") & (data["Reference"]== "Lateral")].groupby("Session").mean()["Lag (ms)"])

mwu = pg.mwu(data[(data["Quantile"]== "Strong ripples") & (data["Reference"]== "Medial")].groupby("Session").mean()["Lag (ms)"],
                 data[(data["Quantile"]== "Strong ripples") & (data["Reference"]== "Lateral")].groupby("Session").mean()["Lag (ms)"])

print(ttest["p-val"], mwu["p-val"])
if ttest["p-val"].values < 0.05:
    axs[0,0].text(.9, .5, "*", transform=axs[0,0].transAxes, fontsize=10, weight="bold");

p_value_ttest_common = '{:0.2e}'.format(ttest["p-val"][0])
axs[0,0].text(.3,.05,"Cohen's d: " + str(round(ttest["cohen-d"].values[0],2)), transform=axs[0,0].transAxes, fontsize=6);

# print(pg.normality(data[data["Quantile"]=="Strong ripples"].groupby(["Session", "Reference"]).mean().reset_index(),
#                    dv="Lag (ms)", group="Reference", method="normaltest"))
ttest_lag = pg.ttest(data[(data["Quantile"]== "Strong ripples") & (data["Reference"]== "Medial")].groupby("Session").mean()["Lag (ms)"],
                 data[(data["Quantile"]== "Strong ripples") & (data["Reference"]== "Lateral")].groupby("Session").mean()["Lag (ms)"])

mwu_lag = pg.mwu(data[(data["Quantile"]== "Strong ripples") & (data["Reference"]== "Medial")].groupby("Session").mean()["Lag (ms)"],
                 data[(data["Quantile"]== "Strong ripples") & (data["Reference"]== "Lateral")].groupby("Session").mean()["Lag (ms)"])


print(ttest_lag["p-val"], mwu_lag["p-val"])
if ttest_lag["p-val"].values < 0.05:
    axs[1,0].text(.9, .5, "*", transform=axs[1,0].transAxes, fontsize=10, weight="bold");

p_value_ttest_strong = '{:0.2e}'.format(ttest["p-val"][0])
axs[1,0].text(.3,.05,"Cohen's d: " + str(round(ttest_lag["cohen-d"].values[0], 2)), transform=axs[1,0].transAxes, fontsize=6);


for n, ytick in enumerate(axs[0, 0].get_yticklabels()):
    ytick.set_color(palette_ML[ytick.get_text()])

for n, ytick in enumerate(axs[1, 0].get_yticklabels()):
    ytick.set_color(palette_ML[ytick.get_text()])

# Strenght comparison

# sns.stripplot(x="Lag (ms)", y="Type", hue="Session",
#               data=tot_diffs[tot_diffs["Lag (ms)"].between(-4,4)], dodge=True, alpha=.25, zorder=1)
g = sns.pointplot(x="Lag (ms)", y="Quantile", hue="Session",
                  data=data[data["Reference"]=="Lateral"], dodge=0.3,
                  join=True,
                  markers="o", scale=0.8, ci=None, palette=sns.color_palette("Greys", 80)[20:], ax=axs[1, 1])
plt.setp(g.collections, alpha=.6)  # for the markers
plt.setp(g.lines, alpha=.6)
g.legend([], [], frameon=False)

gg = sns.pointplot(x="Lag (ms)", y="Quantile",
                   data=data[data["Reference"]=="Lateral"].groupby(["Session", "Quantile"]).mean().reset_index(),
                   join=True,
                   markers="o", scale=1.5, ci=95, palette=palette_timelags, ax=axs[1, 1], order=["Strong ripples", "Common ripples"])
plt.setp(gg.collections, zorder=100)  # for the markers
plt.setp(gg.lines, zorder=100)
axs[1, 1].set_title("Lateral Reference", color=palette_ML["Lateral"])

g = sns.pointplot(x="Lag (ms)", y="Quantile", hue="Session",
                  data=data[data["Reference"]=="Medial"], dodge=0.3,
                  join=True,
                  markers="o", scale=0.8, ci=None, palette=sns.color_palette("Greys", 80)[20:], ax=axs[0, 1])
plt.setp(g.collections, alpha=.6)  # for the markers
plt.setp(g.lines, alpha=.6)
g.legend([], [], frameon=False)

gg = sns.pointplot(x="Lag (ms)", y="Quantile",
                   data=data[data["Reference"]=="Medial"].groupby(["Session", "Quantile"]).mean().reset_index(),
                   join=True,
                   markers="o", scale=1.5, ci=95, palette=palette_timelags, ax=axs[0, 1], order=["Strong ripples", "Common ripples"])
plt.setp(gg.collections, zorder=100);  # for the markers
plt.setp(gg.lines, zorder=100)
axs[0, 1].set_ylabel('')
axs[0, 1].set_title("Medial Reference", color=palette_ML["Medial"])

axs[1, 1].set_ylabel('')


#if non parametric needed use mwu instead of ttest
# print(pg.normality(data[data["Reference"] == "Medial"].groupby(["Session", "Quantile"]).mean().reset_index(),
#                    dv="Lag (ms)", group="Quantile", method="normaltest"))
ttest = pg.ttest(data[(data["Quantile"]== "Strong ripples") & (data["Reference"]== "Medial")].groupby("Session").mean()["Lag (ms)"],
                 data[(data["Quantile"]== "Common ripples") & (data["Reference"]== "Medial")].groupby("Session").mean()["Lag (ms)"])

print(ttest["p-val"], mwu["p-val"])

if ttest["p-val"].values < 0.05:
    axs[0,1].text(.9, .5, "*", transform=axs[0,1].transAxes, fontsize=10, weight="bold");

p_value_ttest_lag_ref_medial = '{:0.2e}'.format(ttest["p-val"][0])
axs[0,1].text(.3,.05,"Cohen's d: " + str(round(ttest["cohen-d"].values[0],2)), transform=axs[0,1].transAxes, fontsize=6);

#print(pg.normality(data[data["Reference"] == "Lateral"].groupby(["Session", "Quantile"]).mean().reset_index(),
#                   dv="Lag (ms)", group="Quantile", method="normaltest"))
ttest = pg.ttest(data[(data["Quantile"]== "Strong ripples") & (data["Reference"]== "Lateral")].groupby("Session").mean()["Lag (ms)"],
                 data[(data["Quantile"]== "Common ripples") & (data["Reference"]== "Lateral")].groupby("Session").mean()["Lag (ms)"])
mwu = pg.mwu(data[(data["Quantile"]== "Strong ripples") & (data["Reference"]== "Lateral")].groupby("Session").mean()["Lag (ms)"],
                 data[(data["Quantile"]== "Common ripples") & (data["Reference"]== "Lateral")].groupby("Session").mean()["Lag (ms)"])

print(ttest["p-val"], mwu["p-val"])
if ttest["p-val"].values < 0.05:
    axs[1,1].text(.9, .5, "*", transform=axs[1,1].transAxes, fontsize=10, weight="bold");
p_value_ttest_lag_ref_lateral = '{:0.2e}'.format(ttest["p-val"][0])
axs[1,1].text(.3,.05,"Cohen's d: " + str(round(ttest["cohen-d"].values[0], 2)), transform=axs[1,1].transAxes, fontsize=6);


axs[0,1].text(.9,.5,"*", transform=axs[0, 1].transAxes, fontsize=10, weight="bold");

#axs[1].text(.8,.5,"*", transform=axs[1].transAxes, fontsize=14, weight="bold");


for n, ytick in enumerate(axs[0, 1].get_yticklabels()):
    ytick.set_color(palette_timelags[ytick.get_text()])

for n, ytick in enumerate(axs[1, 1].get_yticklabels()):
    ytick.set_color(palette_timelags[ytick.get_text()])

axs[0,0].set_xlim(-22, 32)
axs[0,1].set_xlim(-22, 32)
axs[1,0].set_xlim(-22, 32)
axs[1,1].set_xlim(-22, 32)
plt.show()
