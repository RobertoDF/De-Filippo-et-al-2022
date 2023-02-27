import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import numpy as np
import Utils.Style
from Utils.Settings import output_folder_figures_calculations

with open(f"{output_folder_figures_calculations}/temp_data_figure_1.pkl", "rb") as fp:  # Unpickling
    sessions, high_distance, low_distance, ripples_lags, ripples_lags_inverted_reference, ripples_calcs, summary_corrs, distance_tabs = pickle.load(fp)

clip = (-50, 50)
ripples_lags = ripples_lags[ripples_lags["Lag (ms)"].between(clip[0], clip[1])]

fig, axs = plt.subplots(2, figsize=(8, 8))
# sns.stripplot(x="Lag (ms)", y="Type", hue="Session",
#               data=tot_diffs[tot_diffs["Lag (ms)"].between(-4,4)], dodge=True, alpha=.25, zorder=1)
g = sns.pointplot(x="Absolute lag (ms)", y="Type", hue="Session",
                  data=ripples_lags, dodge=0.3,
                  join=True,
                  markers="o", scale=0.8, ci=None, palette=sns.color_palette("Greys",50)[5:20], ax=axs[1])
plt.setp(g.collections, alpha=.6)  # for the markers
plt.setp(g.lines, alpha=.6)
g.legend([], [], frameon=False)

gg = sns.pointplot(x="Absolute lag (ms)", y="Type",
                   data=ripples_lags.groupby(["Session", "Type"]).mean().reset_index(),
                   join=True,
                   markers="o", scale=1.5, ci=95, palette="husl", ax=axs[1])
plt.setp(gg.collections, zorder=100)  # for the markers
plt.setp(gg.lines, zorder=100)

g = sns.pointplot(x="Lag (ms)", y="Type", hue="Session",
                  data=ripples_lags, dodge=0.3,
                  join=True,
                  markers="o", scale=0.8, ci=None, palette=sns.color_palette("Greys", 80)[20:], ax=axs[0])
plt.setp(g.collections, alpha=.6)  # for the markers
plt.setp(g.lines, alpha=.6)
g.legend([], [], frameon=False)

gg = sns.pointplot(x="Lag (ms)", y="Type",
                   data=ripples_lags.groupby(["Session", "Type"]).mean().reset_index(),
                   join=True,
                   markers="o", scale=1.5, ci=95, palette="husl", ax=axs[0])
plt.setp(gg.collections, zorder=100);  # for the markers
plt.setp(gg.lines, zorder=100)
axs[0].set_ylabel('')
axs[1].set_ylabel('')


for n, ytick in enumerate(axs[0].get_yticklabels()):
    ytick.set_color(sns.husl_palette(2)[n])

for n, ytick in enumerate(axs[1].get_yticklabels()):
    ytick.set_color(sns.husl_palette(2)[n])

#print(pg.normality(ripples_lags.groupby(["Session", "Type"]).mean().reset_index(), dv="Lag (ms)", group="Type", method="normaltest"))
ttest = pg.ttest(ripples_lags[ripples_lags["Type"] =="High distance (µm)"].groupby("Session").mean()["Lag (ms)"],
                 ripples_lags[ripples_lags["Type"] =="Low distance (µm)"].groupby("Session").mean()["Lag (ms)"])
mwu = pg.mwu(ripples_lags[ripples_lags["Type"] =="High distance (µm)"].groupby("Session").mean()["Lag (ms)"],
                 ripples_lags[ripples_lags["Type"] =="Low distance (µm)"].groupby("Session").mean()["Lag (ms)"])
print(ttest["p-val"], mwu["p-val"])
if ttest["p-val"].values < 0.05:
    axs[0].text(.9, .5, "*", transform=axs[0].transAxes, fontsize=10, weight="bold");

axs[0].text(.3, .05, "Cohen's d: " + str(round(ttest["cohen-d"].values[0],2)), transform=axs[0].transAxes, fontsize=6);

p_value_ttest_lag = '{:0.2e}'.format(ttest["p-val"][0])

#print(pg.normality(ripples_lags.groupby(["Session", "Type"]).mean().reset_index(), dv="Absolute lag (ms)", group="Type", method="normaltest"))
ttest = pg.ttest(ripples_lags[ripples_lags["Type"] =="High distance (µm)"].groupby("Session").mean()["Absolute lag (ms)"],
                 ripples_lags[ripples_lags["Type"] =="Low distance (µm)"].groupby("Session").mean()["Absolute lag (ms)"])
mwu = pg.mwu(ripples_lags[ripples_lags["Type"] =="High distance (µm)"].groupby("Session").mean()["Absolute lag (ms)"],
                 ripples_lags[ripples_lags["Type"] =="Low distance (µm)"].groupby("Session").mean()["Absolute lag (ms)"])

print(ttest["p-val"], mwu["p-val"])
if ttest["p-val"].values < 0.05:
    axs[1].text(.9, .5, "*", transform=axs[1].transAxes, fontsize=10, weight="bold");

p_value_ttest_abs_lag = '{:0.2e}'.format(ttest["p-val"][0])

axs[1].text(.3, .05, "Cohen's d: " + str(round(ttest["cohen-d"].values[0], 2)), transform=axs[1].transAxes, fontsize=6);


plt.show()


