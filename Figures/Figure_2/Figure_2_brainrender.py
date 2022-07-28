import pickle
import seaborn as sns
from brainrender import Scene, settings
from brainrender.actors import Points, Point
from PIL import Image
import numpy as np
from Utils.Settings import output_folder_figures_calculations

with open(f'{output_folder_figures_calculations}/temp_data_figure_2.pkl', 'rb') as f:
    session_id, session_trajs, columns_to_keep, ripples, real_ripple_summary, lfp_per_probe, \
    ripple_cluster_strong, ripple_cluster_weak, example_session = pickle.load(f)


ripple_cluster = ripple_cluster_strong.sort_values(by="M-L (µm)")
settings.SHADER_STYLE = "cartoon"
settings.SHOW_AXES = False

color_palette = sns.color_palette("flare", 255)

scene = Scene()

ca1 = scene.add_brain_region("CA", hemisphere="left", alpha=0.1)

#scene.slice('sagittal', actors=[ca1], invert=False)

pos_area = ripple_cluster.iloc[1:,:][["A-P (µm)", "D-V (µm)", "L-R (µm)"]].values
print(pos_area)
colors = [color_palette[n] for n in ripple_cluster.iloc[1:, :]["color index"]]

ca1s = Points(
    pos_area,
    radius=200,
    alpha=1,
    colors=colors,
    name=f"Correlations ∫Ripple")

scene.add(ca1s)

pos_area = ripple_cluster.iloc[0,:][["A-P (µm)", "D-V (µm)", "L-R (µm)"]].values
color = color_palette[ripple_cluster.iloc[0,:]["color index"]]

ca1s_source = Point(
    pos_area,
    radius=450,
    alpha=0.5,
    color=color,
    name=f"Correlations ∫Ripple")

scene.add(ca1s_source)
# define camera positions
cam0 = {
     'pos': (-18912, -4980, -28900),
     'viewup': (0, -1, 0),
     'clippingRange': (17189, 58828),
   }

scene.render(zoom=1.0, camera=cam0)
scene.screenshot(name=f"{output_folder_figures_calculations}/Figure_2_temp.png", scale=4)
im = Image.open(f'{output_folder_figures_calculations}/Figure_2_temp.png')
width, height = im.size
im = im.crop((2250, 1100, width - 2425, height - 1000))
im.save(f'{output_folder_figures_calculations}/Figure_2_brainrender_crop.png')



