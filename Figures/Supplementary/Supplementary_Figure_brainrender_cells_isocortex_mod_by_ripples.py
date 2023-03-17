
from PIL import Image
from brainrender import Scene, settings
from brainrender.actors import Points
import dill
from Utils.Settings import output_folder_supplementary, output_folder_figures_calculations
from Utils.Style import palette_ML
from rich import print
from Utils.Utils import rgb2hex
from brainrender import Scene
from brainrender.video import VideoMaker


from rich import print



# Create an instance of video maker




settings.SHOW_AXES = False
with open(f"{output_folder_figures_calculations}/temp_data_figure_5.pkl", 'rb') as f:
    summary_units_df_sub = dill.load(f)

scene = Scene()

data = summary_units_df_sub[(summary_units_df_sub["Parent brain region"]=="Isocortex") &
                    (summary_units_df_sub['Ripple modulation (0-50 ms) lateral'] > .5) &
                            (summary_units_df_sub['Ripple modulation (0-50 ms) medial'] < .5) ]

ca1 = scene.add_brain_region("CA1", alpha=0.15, color="#754668")

# Add to scene
scene.add(Points(data[["A-P", "D-V", "L-R"]].values, name="CELLS", radius=50, alpha=.5, colors=rgb2hex(palette_ML["Lateral"])))

data = summary_units_df_sub[(summary_units_df_sub["Parent brain region"]=="Isocortex") &
                    (summary_units_df_sub['Ripple modulation (0-50 ms) medial']>.5) &
                            (summary_units_df_sub['Ripple modulation (0-50 ms) lateral'] < .5)
                            ]


scene.add(Points(data[["A-P", "D-V", "L-R"]].values, name="CELLS", radius=50,  alpha=.5, colors=rgb2hex(palette_ML["Medial"])))

data = summary_units_df_sub[(summary_units_df_sub["Parent brain region"]=="Isocortex") &
                    (summary_units_df_sub['Ripple modulation (0-50 ms) medial']>.5) &
                            (summary_units_df_sub['Ripple modulation (0-50 ms) lateral'] > .5)
                            ]


scene.add(Points(data[["A-P", "D-V", "L-R"]].values, name="CELLS", radius=50,  alpha=.5, colors="r"))
#ca3 = scene.add_brain_region("CA3", alpha=0.15, color="#FF4365")


# render
cam0:{
     'pos': (-7265, -940, -18793),
     'viewup': (0, -1, 0),
     'clippingRange': (1407, 42647),
     'focalPoint': (7830, 4296, -5694),
     'distance': 20661,
   }

vm = VideoMaker(scene, output_folder_supplementary, "Supplementary_Figure_brainrender_cells")

# make a video with the custom make frame function
# this just rotates the scene
vm.make_video(azimuth=2, duration=10, fps=15)

"""
scene.render(zoom=1.0, camera=cam0)
scene.screenshot(name=f"{output_folder_supplementary}/Supplementary_Figure_brainrender_cells.png", scale=4)
im = Image.open(f'{output_folder_supplementary}/Supplementary_Figure_brainrender_cells.png')
width, height = im.size
im = im.crop((2000, 1200, width - 2400, height - 1200))
im.save(f'{output_folder_supplementary}/Supplementary_Figure_brainrender_cells.png')
print("Brainrender saved")"""