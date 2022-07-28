import dill
import seaborn as sns
from brainrender import Scene, settings
from brainrender.actors import Points
from PIL import Image
import pandas as pd

from Utils.Settings import output_folder_figures_calculations, output_folder_calculations

with open(f'{output_folder_calculations}/trajectories_spatial_infos.pkl', 'rb') as f:
    spatial_info_medial, spatial_info_lateral, spatial_info_center = dill.load(f)

medial_source = pd.DataFrame([q[0] for q in spatial_info_medial])
medial_else = pd.concat([q[1] for q in spatial_info_medial])
lateral_source = pd.DataFrame([q[0] for q in spatial_info_lateral])
lateral_else = pd.concat([q[1] for q in spatial_info_lateral])
center_source = pd.DataFrame([q[0] for q in spatial_info_center])
center_else = pd.concat([q[1] for q in spatial_info_center])

settings.SHADER_STYLE = "cartoon"
settings.SHOW_AXES = False

color_palette = sns.color_palette("flare", 255)

scene = Scene()

ca1 = scene.add_brain_region("CA", hemisphere="left", alpha=0.1)

scene.slice('sagittal', invert=True)

source = Points(
    medial_source[["A-P (µm)", "D-V (µm)", "L-R (µm)"]].values,
    radius=100,
    alpha=.6,
    colors="#F03A47",
    name=f"Correlations ∫Ripple")

scene.add(source)

other = Points(
    medial_else[["A-P (µm)", "D-V (µm)", "L-R (µm)"]].values,
    radius=100,
    alpha=.6,
    colors="#353B3C",
    name=f"Correlations ∫Ripple")

scene.add(other)


# define camera positions
cam0 = {
      'pos': (-25840, -3865, -14489),
     'viewup': (0, -1, 0),
     'clippingRange': (18338, 53579),
   }

scene.render(zoom=1.0, camera=cam0)
scene.screenshot(name=f"{output_folder_figures_calculations}/Figure_2_temp_medial.png", scale=4)
im = Image.open(f'{output_folder_figures_calculations}/Figure_2_temp_medial.png')
width, height = im.size
im = im.crop((2750, 1000, width - 3000, height - 900))
im.save(f'{output_folder_figures_calculations}/Figure_2_brainrender_medial_crop.png')



