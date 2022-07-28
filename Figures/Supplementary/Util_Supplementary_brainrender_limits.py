from brainrender import Scene, settings
from brainrender.actors import Points
from PIL import Image
import pandas as pd
import dill
import seaborn as sns
from Utils.Utils import clean_ripples_calculations
from Utils.Settings import output_folder_calculations, output_folder_figures_calculations, var_thr, output_folder_supplementary


color_palette = sns.color_palette("flare", 255)


with open(f'{output_folder_calculations}/clean_ripples_calculations.pkl', 'rb') as f:
    ripples_calcs = dill.load(f)


input_rip = []
CA1_locations = []
for session_id in ripples_calcs.keys():
    ripples = ripples_calcs[session_id][3].copy()
    ripples = ripples.groupby("Probe number-area").filter(lambda group: group["∫Ripple"].var() > var_thr)
    input_rip.append(ripples.groupby("Probe number-area").mean()["L-R (µm)"])
    CA1_locations.append(ripples.groupby("Probe number-area").mean()[["D-V (µm)", "A-P (µm)", "L-R (µm)"]])

CA1_locations = pd.concat(CA1_locations)
lr_space = pd.concat(input_rip)

medial_lim = lr_space.quantile(.33333)
lateral_lim = lr_space.quantile(.666666)

settings.SHOW_AXES = False

scene = Scene()


ca1_l = scene.add_brain_region("CA", alpha=0.5, color=color_palette[254], force=True)

plane = scene.atlas.get_plane(pos=[-7970.65685714, -3228.97190177, -lateral_lim], norm=(0,0,-1))

scene.slice(plane, actors=[ca1_l])

ca1_c = scene.add_brain_region("CA", alpha=0.5, color=color_palette[int(254/2)], force=True)

plane = scene.atlas.get_plane(pos=[-7970.65685714, -3228.97190177, -lateral_lim], norm=(0,0,1))

scene.slice(plane, actors=[ca1_c])

plane = scene.atlas.get_plane(pos=[-7970.65685714, -3228.97190177, -medial_lim], norm=(0,0,-1))

scene.slice(plane, actors=[ca1_c])

ca1_m = scene.add_brain_region("CA", hemisphere="left", alpha=0.5, color=color_palette[0], force=True)

plane = scene.atlas.get_plane(pos=[-7970.65685714, -3228.97190177, -medial_lim], norm=(0,0,1))

scene.slice(plane, actors=[ca1_m])

plane = scene.atlas.get_plane(pos=[7970.65685714, 3678.97190177, medial_lim], norm=(0,-1,0))

scene.slice(plane, actors=[ca1_m, ca1_l, ca1_c])

ca1_excluded = scene.add_brain_region("CA", alpha=0.5, color=(.6,.6,.6), hemisphere="left", force=True)

plane = scene.atlas.get_plane(pos=[7970.65685714, 3678.97190177, medial_lim], norm=(0,1,0))

scene.slice(plane, actors=[ca1_excluded])
# plane = scene.atlas.get_plane(pos=[3970.65685714, 7228.97190177, medial_lim], plane="sagittal")
# scene.slice(plane, actors=[mos, ca1])
# define camera positions

source = Points(
    CA1_locations[["A-P (µm)", "D-V (µm)", "L-R (µm)"]].values,
    radius=30,
    alpha=1,
    colors="#353B3C",
    name=f"Correlations ∫Ripple")

scene.add(source)

cam0 = {
      'pos': (-25840, -3865, -14489),
     'viewup': (0, -1, 0),
     'clippingRange': (18338, 53579),
   }

scene.render(zoom=1.0, camera=cam0)
scene.screenshot(name=f"{output_folder_figures_calculations}/brainrender_hippocampal_sectors.png", scale=4)
im = Image.open(f'{output_folder_figures_calculations}/brainrender_hippocampal_sectors.png')
width, height = im.size
im = im.crop((2200, 1000, width - 2200, height - 1000))
im.save(f'{output_folder_figures_calculations}/brainrender_hippocampal_sectors_crop.png')



