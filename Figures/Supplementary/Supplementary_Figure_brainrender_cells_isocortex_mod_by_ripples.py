
from PIL import Image
from brainrender import Scene, settings
from brainrender.actors import Points
import dill
from Utils.Settings import output_folder_figures_calculations

from rich import print


settings.SHOW_AXES = False
with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)

scene = Scene()

data = summary_units_df_sub[(summary_units_df_sub["Parent brain region"]=="Isocortex") &
                    (summary_units_df_sub['Ripple modulation (0-50 ms) lateral']>2)]

ca1 = scene.add_brain_region("CA1", alpha=0.15, color="#754668")

# Add to scene
scene.add(Points(data[[ "A-P", "D-V", "L-R"]].values, name="CELLS", colors="r"))



#ca3 = scene.add_brain_region("CA3", alpha=0.15, color="#FF4365")


# render
cam0 = {
     'pos': (-18912, -4980, -28900),
     'viewup': (0, -1, 0),
     'clippingRange': (17189, 58828),
   }

scene.render(zoom=1.0, camera=cam0)
#scene.screenshot(name=f"{output_folder_figures_calculations}/Figure_5_brainrender_cells.png", scale=4)
#im = Image.open(f'{output_folder_figures_calculations}/Figure_5_brainrender_cells.png')
#width, height = im.size
#im = im.crop((2000, 1200, width - 2400, height - 1200))
#im.save(f'{output_folder_figures_calculations}/Figure_5_brainrender_cells.png')
print("Brainrender saved")