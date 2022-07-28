from Utils.Settings import output_folder_figures_calculations, output_folder_calculations
from tqdm import tqdm
from Utils.Utils import acronym_structure_id_map
import dill
import pandas as pd
from scipy.stats import zscore
from Utils.Utils import clean_ripples_calculations, is_outlier, acronym_structure_path_map,  \
    acronym_structure_graph_order_map,  summary_structures_finer

with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)


out = []
for area in summary_structures_finer["acronym"]:
    if area not in {"DG"}:# keep DG , (this set is used inverted, exclude means include)
        out.extend(acronym_structure_path_map.get(area))
ids_fine_general_areas = set(out)

out = []
out2 = []
out3 = []
out4 = []
positions = []
out5 = []
positions2 = []
out6 = []
for session_id, sel in tqdm(ripples_calcs.items()):
    selected_probe = sel[5].copy()
    ripple_power = sel[0][0].copy()
    pos_area = sel[0][1].copy()
    neg_area = sel[0][2].copy()
    data_area = -neg_area + pos_area

    out.append(ripple_power.mean().droplevel(0).reset_index())
    out2.append(data_area.mean().droplevel(0).reset_index())
    out3.append(ripple_power.var().droplevel(0).reset_index())
    out4.append(data_area.var().droplevel(0).reset_index())
    positions.append(sel[1][["Probe number","M-L (µm)", "A-P (µm)", "D-V (µm)"]])
    out5.append(ripple_power[selected_probe, "CA1"].mean())
    positions2.append(
        sel[1][(sel[1]["Area"] == "CA1") & (sel[1]["Probe number"] == selected_probe)][["Probe number", "M-L (µm)", "D-V (µm)", "A-P (µm)"]])
    out6.append([session_id] * ripple_power.shape[1])

positions = pd.concat(positions).reset_index(drop=True).infer_objects()
positions2 = pd.concat(positions2).reset_index(drop=True).infer_objects()
sess = pd.Series([item for sublist in out6 for item in sublist], name="Session")
summary_table = pd.concat([pd.concat(out), pd.concat(out2).iloc[:, 1], pd.concat(out3).iloc[:, 1], pd.concat(out4).iloc[:, 1]],
                 axis=1)

summary_table.reset_index(drop=True, inplace=True)
summary_table.columns = ["Area", "μ(∫Ripple)", "μ(RIVD)", "$σ^2$(∫Ripple)", "$σ^2$(RIVD)"]
summary_table = pd.concat([summary_table, positions, sess], axis=1)

summary_table['Count'] = summary_table.groupby('Area')['Area'].transform('count')
summary_table["Area id"] = [acronym_structure_id_map.get(area.split("-")[0]) for area in summary_table["Area"]]
summary_table = summary_table[~summary_table["Area id"].isin(ids_fine_general_areas)]
summary_table["μ(Z-scored ∫Ripple)"] = summary_table.groupby("Session")["μ(∫Ripple)"].transform(lambda x: zscore(x, ddof=1))
summary_table["μ(Z-scored RIVD)"] = summary_table.groupby("Session")["μ(RIVD)"].transform(lambda x: zscore(x, ddof=1))
summary_table["Graph order"] = [acronym_structure_graph_order_map.get(area.split("-")[0]) for area in summary_table["Area"]]
summary_table = summary_table.loc[~summary_table.index.isin(summary_table[summary_table["μ(RIVD)"] == 0].index),:]  # Some areas return zeros in both μ(∫Ripple) and μ(RIVD)
summary_table = summary_table[~summary_table.groupby('Area')['μ(∫Ripple)'].apply(is_outlier)]

session_id = 791319847

ripple_power = ripples_calcs[session_id][0][0].copy()
pos_area = ripples_calcs[session_id][0][1].copy()
neg_area = ripples_calcs[session_id][0][2].copy()
spatial_info = ripples_calcs[session_id][1].copy()
ripples = ripples_calcs[session_id][3].copy()
data_area = -neg_area + pos_area

session_summary = pd.concat([ripple_power.droplevel(0,axis=1).stack().reset_index(), data_area.droplevel(0,axis=1).stack().reset_index().iloc[:,-1:]], axis=1)
session_summary.columns = ["Time (s)", "Area", "∫Ripple", "RIVD"]
session_summary["Z-scored ∫Ripple"] = zscore(session_summary["∫Ripple"])
session_summary["Z-scored RIVD"] = zscore(session_summary["RIVD"])

high_var_areas_sess = session_summary.groupby("Area").std()[["Z-scored ∫Ripple", "Z-scored RIVD"]].sum(axis=1).sort_values(ascending=False)
high_var_areas_sess = high_var_areas_sess[high_var_areas_sess>high_var_areas_sess.std()*2.5].index


with open(f"{output_folder_figures_calculations}/temp_data_supp_figure_1.pkl", "wb") as fp:
    dill.dump([data_area, ripple_power, session_id, session_summary, summary_table, ripples], fp)