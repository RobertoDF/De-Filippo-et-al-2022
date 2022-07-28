import pickle
import seaborn as sns
from brainrender import Scene, settings
from brainrender.actors import Points, Point
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from Utils.Settings import output_folder_figures_calculations

with open(f'{output_folder_figures_calculations}/temp_data_figure_4.pkl', 'rb') as f:
    space_sub_spike_times, target_area, units, field_to_use_to_compare, \
    session_id_example, lfp, lfp_per_probe, \
    ripple_cluster_lateral_seed, ripple_cluster_medial_seed, source_area, ripples, \
    tot_summary_early, summary_fraction_active_clusters_per_ripples_early, \
    summary_fraction_active_clusters_per_ripples_early_by_neuron_type, \
    tot_summary_late, summary_fraction_active_clusters_per_ripples_late, \
    summary_fraction_active_clusters_per_ripples_late_by_neuron_type, \
    tot_summary, summary_fraction_active_clusters_per_ripples, \
    summary_fraction_active_clusters_per_ripples_by_neuron_type = pickle.load(f)


ripple_cluster = ripples.groupby("Probe number-area").mean().sort_values("L-R (µm)").reset_index()

scaler = MinMaxScaler()
colors_idx = scaler.fit_transform(ripple_cluster["L-R (µm)"].values.reshape(-1,1)) * 254
colors_idx = colors_idx.astype(int)
ripple_cluster["color index"] = colors_idx

settings.SHADER_STYLE = "cartoon"
settings.SHOW_AXES = False

color_palette = sns.color_palette("flare", 255)

scene = Scene()

ca1 = scene.add_brain_region("CA", hemisphere="left", alpha=0.1)

scene.slice('sagittal',  invert=True)

pos_area = ripple_cluster[ripple_cluster["Probe number-area"]!=source_area][["A-P (µm)", "D-V (µm)", "L-R (µm)"]].values
colors = [color_palette[n] for n in ripple_cluster[ripple_cluster["Probe number-area"]!=source_area]["color index"]]

ca1s = Points(
    pos_area,
    radius=200,
    alpha=1,
    colors=colors,
    name=f"Correlations ∫Ripple")

scene.add(ca1s)

pos_area = ripple_cluster[ripple_cluster["Probe number-area"]==source_area][["A-P (µm)", "D-V (µm)", "L-R (µm)"]].values[0]
color = color_palette[ripple_cluster[ripple_cluster["Probe number-area"]==source_area]["color index"].values[0]]

ca1s_source = Point(
    pos_area,
    radius=450,
    alpha=0.5,
    color=color,
    name=f"Correlations ∫Ripple")

scene.add(ca1s_source)
# define camera positions
cam0 = {
      'pos': (-25840, -3865, -14489),
     'viewup': (0, -1, 0),
     'clippingRange': (18338, 53579),
   }

scene.render(zoom=1.0, camera=cam0)
scene.screenshot(name=f"{output_folder_figures_calculations}/Figure_2_temp.png", scale=4)
im = Image.open(f'{output_folder_figures_calculations}/Figure_2_temp.png')
width, height = im.size
im = im.crop((2250, 1100, width - 2425, height - 1000))
im.save(f'{output_folder_figures_calculations}/Figure_4_brainrender_crop.png')



