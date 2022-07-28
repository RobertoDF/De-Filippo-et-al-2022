from brainrender import Scene, settings
from brainrender.actors import Points
from PIL import Image
import pandas as pd
import dill
import seaborn as sns
from Utils.Utils import clean_ripples_calculations
from Utils.Settings import output_folder_calculations, output_folder_figures_calculations, output_folder_supplementary


settings.SHOW_AXES = False

scene = Scene()

ca = scene.add_brain_region("CA", alpha=0.5, hemisphere="left")

plane = scene.atlas.get_plane(pos=[7970.65685714, 3678.97190177, 7772.97056], norm=(0,-1,0))

scene.slice(plane, actors=[ca])

ca1_excluded = scene.add_brain_region("CA", alpha=0.5, color=(.6,.6,.6), hemisphere="left", force=True)

plane = scene.atlas.get_plane(pos=[7970.65685714, 3678.97190177, 7772.97056], norm=(0,1,0))

scene.slice(plane, actors=[ca1_excluded])

cam0 = {
      'pos': (-20965, -10051, -21271),
     'viewup': (0, -1, 0),
     'clippingRange': (16153, 57828),
   }

scene.render(zoom=1.0, camera=cam0)
scene.screenshot(name=f"{output_folder_figures_calculations}/brainrender_abstract.png", scale=4)
im = Image.open(f'{output_folder_figures_calculations}/brainrender_abstract.png')
width, height = im.size
im = im.crop((2200, 900, width - 2200, height - 850))
im.save(f'{output_folder_figures_calculations}/brainrender_abstract_crop.png')



