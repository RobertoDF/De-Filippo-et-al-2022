from tqdm import tqdm
from scipy.stats import zscore
from Utils.Settings import output_folder_calculations, output_folder_supplementary, var_thr, Adapt_for_Nature_style
from Utils.Utils import Naturize
import dill
from Utils.Utils import clean_ripples_calculations, corrfunc
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import Utils.Style


with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)


out = []
for session_id, sel in tqdm(ripples_calcs.items()):
    ripples = ripples_calcs[session_id][3].copy()
    ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)

    ripples["Z-scored ∫Ripple"] = ripples["∫Ripple"].transform(lambda x: zscore(x, ddof=1))
    ripples["Z-scored amplitude (mV)"] = ripples["Amplitude (mV)"].transform(lambda x: zscore(x, ddof=1))
    ripples["Session"] = session_id

    out.append(ripples.groupby("Probe number").mean())

data = pd.concat(out)
data["Session"] = data["Session"].astype("category")
data.reset_index(inplace=True)

g = sns.pairplot(data=data, x_vars=["M-L (µm)", "A-P (µm)", "D-V (µm)"], y_vars=["Z-scored ∫Ripple", "∫Ripple"], kind="reg", plot_kws=dict(scatter_kws={"alpha": 0.6, "s":20, "color":(0.4,0.4,0.4)}, line_kws={"alpha": 0.8,  "color":"#D7263D", "linestyle": "--"}))
g.map_upper(corrfunc, y_pos=.8)
g.map_offdiag(corrfunc, y_pos=.8)

#plt.show()
plt.savefig(f"{output_folder_supplementary}/Supplementary_Figure_6", dpi=300)
