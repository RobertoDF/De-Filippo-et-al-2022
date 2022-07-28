import pickle
from brainrender import Scene, settings
from brainrender.actors import Points
from PIL import Image
import numpy as np
from Utils.Settings import output_folder_figures_calculations

with open(f"{output_folder_figures_calculations}/temp_data_figure_1.pkl", "rb") as fp:  # Unpickling
    sessions, high_distance, low_distance, ripples_lags, ripples_lags_inverted_reference,  ripples_calcs, summary_corrs = pickle.load(fp)

settings.SHADER_STYLE = "cartoon"
settings.SHOW_AXES = False

scene = Scene()

pos_area = high_distance.groupby(["Session", "Probe number"])[["A-P (µm)", "D-V (µm)", "L-R (µm)"]].mean().values

scene.add_brain_region("CA", alpha=0.1)

colors = ["#73323B", "#DE9BA4"] * (int(np.ceil(pos_area.shape[0]/2)))

probe_sphere = Points(
    pos_area,
    radius=140,
    alpha=0.5,
    colors=colors,
    name=f"Correlations ∫Ripple")

scene.add(probe_sphere)


# define camera positions
cam0 = {
     'pos': (-18912, -4980, -28900),
     'viewup': (0, -1, 0),
     'clippingRange': (17189, 58828),
   }

scene.render(zoom=1.0, camera=cam0)
scene.screenshot(name=f"{output_folder_figures_calculations}/Figure_1_temp_high_distance_brainspace.png", scale=4)
im = Image.open(f'{output_folder_figures_calculations}/Figure_1_temp_high_distance_brainspace.png')
width, height = im.size
im = im.crop((2200, 1000, width - 2400, height - 1000))
im.save(f'{output_folder_figures_calculations}/Figure_1_brainrender_high_dist.png')
print("Brainrender saved")



