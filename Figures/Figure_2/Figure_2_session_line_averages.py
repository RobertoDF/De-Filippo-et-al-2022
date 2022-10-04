import dill
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import Utils.Style
from Utils.Style import palette_timelags
from Utils.Settings import output_folder_figures_calculations

with open(f"{output_folder_figures_calculations}/temp_data_figure_2.pkl", 'rb') as f:
    session_id, session_trajs, columns_to_keep, ripples, real_ripple_summary, \
    lfp_per_probe, ripple_cluster_strong, ripple_cluster_weak, example_session = dill.load(f)

fig, axs = plt.subplots(1, 1, figsize=(8, 4))

sns.lineplot(data=example_session[example_session["Type"] != "Total ripples"], x="M-L (µm)", y="Lag (ms)", hue="Type",
             palette=palette_timelags)

only_lags = real_ripple_summary[real_ripple_summary["∫Ripple"] > real_ripple_summary["∫Ripple"].quantile(.9)][columns_to_keep]*1000
y = only_lags.mean()
error = only_lags.std()/np.sqrt(real_ripple_summary[real_ripple_summary["∫Ripple"] > real_ripple_summary["∫Ripple"].quantile(.9)].shape[0])
axs.fill_between(example_session["M-L (µm)"].unique(), y-error, y+error, alpha=.3, color = palette_timelags["Strong ripples"])


only_lags = real_ripple_summary[real_ripple_summary["∫Ripple"] < real_ripple_summary["∫Ripple"].quantile(.9)][columns_to_keep]*1000
y = only_lags.mean()
error = only_lags.std()/np.sqrt(real_ripple_summary[real_ripple_summary["∫Ripple"] > real_ripple_summary["∫Ripple"].quantile(.9)].shape[0])
axs.fill_between(example_session["M-L (µm)"].unique(), y-error, y+error, alpha=.3, color = palette_timelags["Common ripples"])

# only_lags = real_ripple_summary[columns_to_keep]*1000
# y = only_lags.mean()
# error = only_lags.std()/np.sqrt(real_ripple_summary[real_ripple_summary["∫Ripple"] > real_ripple_summary["∫Ripple"].quantile(.9)].shape[0])
# axs.fill_between(example_session["M-L (µm)"].unique(), y-error, y+error, alpha=.3, color = palette_timelags["Total ripples"])
plt.legend(loc='upper left', title="")
plt.show()
