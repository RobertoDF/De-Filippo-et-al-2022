from Utils.brainrender_utils import brainrender_probes
import pickle
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from Utils.Settings import neuropixel_dataset, output_folder_figures_calculations

manifest_path = f"{neuropixel_dataset}/manifest.json"
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)


with open(f"{output_folder_figures_calculations}/temp_data_supp_figure_1.pkl", "rb") as fp:  # Unpickling
    _, _, session_id, _, _, _ = pickle.load(fp)

session = cache.get_session_data(session_id)

#if not exists(f'home/roberto/Github/Allen_Institute_Neuropixel/pictures/probes_{session_id}.png'):
plot_type = "export" #"video" or "static" or "export"
color_type = "area"
show_labels = False
brainrender_probes(session, plot_type, color_type, show_labels, session_id, output_folder_figures_calculations)
