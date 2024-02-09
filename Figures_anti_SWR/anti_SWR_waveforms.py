import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Utils.Settings import window_spike_hist, output_folder_calculations, neuropixel_dataset, var_thr, minimum_ripples_count_spike_analysis, minimum_ripples_count_generated_in_lateral_or_medial_spike_analysis
import dill
from Utils.Utils import acronym_color_map
from Utils.Settings import output_folder

palette_anti_SWR={True: "#A268E4", False: "#454545"}

with open(f'{output_folder_calculations}/CA1_waveforms_anti.pkl', 'rb') as f:
    exc, exc_anti, inh, inh_anti, time = dill.load(f)
fig, axs = plt.subplots(2,2)
pd.DataFrame(np.vstack(inh), columns=time*1000).T.plot(legend=False, color=palette_anti_SWR[False], alpha=.1, ax=axs[0,0])
pd.DataFrame(np.vstack(inh_anti), columns=time*1000).T.plot(legend=False, color=palette_anti_SWR[True], alpha=.5, ax=axs[0,0])
axs[0,0].set_xlim((0.25,1.5))

x = time*1000
y = pd.DataFrame(np.vstack(inh), columns=time*1000).mean()
error = pd.DataFrame(np.vstack(inh), columns=time*1000).sem()

axs[0,1].fill_between(x, y-error, y+error, color=palette_anti_SWR[False], alpha=.5)
axs[0,1].plot(x,y, color=palette_anti_SWR[False], label="Standard")

y = pd.DataFrame(np.vstack(inh_anti), columns=time*1000).mean()
error = pd.DataFrame(np.vstack(inh_anti), columns=time*1000).sem()

axs[0,1].fill_between(x, y-error, y+error, color=palette_anti_SWR[True], alpha=.5)
axs[0,1].plot(x,y, color=palette_anti_SWR[True], label="Anti-SWR")
axs[0,1].set_xlim((0.25,1.5))
axs[0,1].legend()
axs[0,1].set_title("Putative inh CA1")


pd.DataFrame(np.vstack(exc), columns=time*1000).T.plot(legend=False, color=palette_anti_SWR[False], alpha=.1, ax=axs[1,0])
pd.DataFrame(np.vstack(exc_anti), columns=time*1000).T.plot(legend=False, color=palette_anti_SWR[True], alpha=.5, ax=axs[1,0])
axs[1,0].set_xlim((0.25,1.5))

x = time*1000
y = pd.DataFrame(np.vstack(exc), columns=time*1000).mean()
error = pd.DataFrame(np.vstack(exc), columns=time*1000).sem()

axs[1,1].fill_between(x, y-error, y+error, color=palette_anti_SWR[False], alpha=.5)
axs[1,1].plot(x,y, color=palette_anti_SWR[False], label="Standard")

y = pd.DataFrame(np.vstack(exc_anti), columns=time*1000).mean()
error = pd.DataFrame(np.vstack(exc_anti), columns=time*1000).sem()

axs[1,1].fill_between(x, y-error, y+error, color=palette_anti_SWR[True], alpha=.5)
axs[1,1].plot(x,y, color=palette_anti_SWR[True], label="Anti-SWR")
axs[1,1].set_xlim((0.25,1.5))
axs[1,1].legend()
axs[1,1].set_title("Putative exc CA1")

#plt.show()

plt.savefig(f"{output_folder}/waveforms", dpi=300)