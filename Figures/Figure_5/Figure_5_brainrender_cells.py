
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


#ca1 = scene.add_brain_region("CA1", alpha=0.15, color="#754668")

# Add to scene
scene.add(Points(summary_units_df_sub[summary_units_df_sub["Brain region"]=="CA1"]
                 [["A-P", "D-V", "L-R"]].values, name="CELLS", colors="#754668"))


#ca3 = scene.add_brain_region("CA3", alpha=0.15, color="#FF4365")

# Add to scene
scene.add(Points(summary_units_df_sub[summary_units_df_sub["Brain region"]=="CA3"]
                 [["A-P", "D-V", "L-R"]].values, name="CELLS", colors="#FF4365"))


#ca3 = scene.add_brain_region("DG", alpha=0.15, color="#2292A4")

# Add to scene
scene.add(Points(summary_units_df_sub[summary_units_df_sub["Brain region"]=="DG"]
                 [["A-P", "D-V", "L-R"]].values, name="CELLS", colors="#2292A4"))


#ProS = scene.add_brain_region("ProS", alpha=0.15, color="#EAC435")

# Add to scene
scene.add(Points(summary_units_df_sub[summary_units_df_sub["Brain region"]=="ProS"]
                 [["A-P", "D-V", "L-R"]].values, name="CELLS", colors="#EAC435"))

#SUB = scene.add_brain_region("SUB", alpha=0.15, color="#04080F")

# Add to scene
scene.add(Points(summary_units_df_sub[summary_units_df_sub["Brain region"]=="SUB"]
                 [["A-P", "D-V", "L-R"]].values, name="CELLS", colors="#04080F"))

# render
cam0 = {
     'pos': (-18912, -4980, -28900),
     'viewup': (0, -1, 0),
     'clippingRange': (17189, 58828),
   }

scene.render(zoom=1.0, camera=cam0)
scene.screenshot(name=f"{output_folder_figures_calculations}/Figure_5_brainrender_cells.png", scale=4)
im = Image.open(f'{output_folder_figures_calculations}/Figure_5_brainrender_cells.png')
width, height = im.size
im = im.crop((2000, 1200, width - 2400, height - 1200))
im.save(f'{output_folder_figures_calculations}/Figure_5_brainrender_cells.png')
print("Brainrender saved")