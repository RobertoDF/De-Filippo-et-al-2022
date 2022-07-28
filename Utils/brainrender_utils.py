from brainrender import Scene, Animation, settings
from brainrender.actors import Points
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import numpy as np
import pandas as pd
from Utils.Utils import acronym_color_map
from PIL import Image
from Utils.Settings import output_folder_figures_calculations

def brainrender_probes(session, plot_type, color_type, show_labels, session_id, output_directory):
    settings.SHOW_AXES = False
    settings.WHOLE_SCREEN = True
    settings.SHADER_STYLE = "cartoon"  # affects the look of rendered brain regions: [cartoon, metallic, plastic, shiny, glossy]

    channels = session.channels

    probe_ids = channels["probe_id"].unique()

    scene = Scene(inset=False, screenshots_folder="pictures/")

    alpha = 0.9

    # scene.add_brain_region("HIP", hemisphere="right", alpha=0.1)
    # scene.add_brain_region("TH", hemisphere="right", alpha=0.15)
    # scene.add_brain_region("VIS", hemisphere="right", alpha=0.15)
    #scene.add_brain_region("MB", alpha=0.1)
    scene.add_brain_region("CA", hemisphere="left", alpha=0.15)

    if color_type == "area":
        colors = iter(plt.cm.Dark2(np.linspace(0, 1, 8)))
        for n, probe_id in enumerate(probe_ids):
            color_label = rgb2hex(next(colors))
            probe = channels[channels["probe_id"] == probe_id].dropna()
            if probe.shape[0] > 0:
                for area in probe["ecephys_structure_acronym"].unique():
                    if pd.isna(area):
                        color = "#adadac"
                        probe_spheres = Points(
                            probe[pd.isna(probe["ecephys_structure_acronym"])][
                                ["anterior_posterior_ccf_coordinate", "dorsal_ventral_ccf_coordinate",
                                 "left_right_ccf_coordinate"]].values,
                            radius=40,
                            alpha=alpha,
                            colors=color)

                        if show_labels:
                            scene.add_label(probe_spheres, f"probe {n}", color=color_label, size=200, radius=200,
                                            xoffset=-4, yoffset=1000, zoffset=500)
                        else:
                            scene.add_label(probe_spheres, f"probe {n}", color=color_label, size=0, radius=200,
                                            xoffset=-4, yoffset=1000, zoffset=500)

                    else:
                        color = rgb2hex([x / 255 for x in acronym_color_map.get(area)])
                        probe_spheres = Points(
                            probe[probe["ecephys_structure_acronym"] == area][
                                ["anterior_posterior_ccf_coordinate", "dorsal_ventral_ccf_coordinate",
                                 "left_right_ccf_coordinate"]].values,
                            radius=40,
                            alpha=alpha,
                            colors=color)


                    scene.add(probe_spheres, names="probe")

                if show_labels:
                    scene.add_label(probe_spheres, f"probe {n}", color=color_label, size=200, radius=200,
                                    xoffset=-4, yoffset=1000, zoffset=500)
                else:
                    scene.add_label(probe_spheres, f"probe {n}", color=color_label, size=0, radius=200,
                                    xoffset=-4, yoffset=1000, zoffset=500)



    else:
        colors = iter(plt.cm.Dark2(np.linspace(0, 1, 8)))
        for probe_id in probe_ids:
            color = rgb2hex(next(colors))
            probe = channels[channels["probe_id"] == probe_id]
            probe_spheres = Points(
                probe[["anterior_posterior_ccf_coordinate", "dorsal_ventral_ccf_coordinate",
                     "left_right_ccf_coordinate"]].values,
                radius=40,
                alpha=alpha,
                colors=color)

            scene.add(probe_spheres, names="probe")
            #scene.add_label(probe_spheres, "probe " + str(n),  size=200, radius=50)



    # define camera positions
    cam0 = {
         'pos': (10302, -8741, -26605),
        'viewup': (0, -1, 1),
        'clippingRange': (10037, 42981),
        'focalPoint': (7370, 4302, -5479),
        'distance': 24999,
    }
    cam1 = {
        'pos': (7467, -7509, -22647),
        'viewup': (0, -1, 1),
        'clippingRange': (7235, 36995),
        'distance': 20661,
    }
    cam2 = {
        'pos': (9035, 6163, -26235),
        'viewup': (0, -1, 0),
        'clippingRange': (8844, 35854),
        'distance': 20661,
    }
    cam3 = {
        'pos': (8198, -14065, -15161),
        'viewup': (0, 0, 1),
        'clippingRange': (8538, 35130),
        'distance': 20661,
    }

    if plot_type == "static":
        # render
        # scene.content
        scene.render(zoom=1.0, camera=cam0)
    elif plot_type == "export":
        scene.render(zoom=1.0, camera=cam0)
        scene.screenshot(name=f"{output_folder_figures_calculations}/probes_" + str(session_id), scale=4)
        im = Image.open(f"{output_folder_figures_calculations}/probes_{session_id}.png")
        width, height = im.size
        im = im.crop((1200, 600, width - 1500, height - 400))# first number sets upper limit,
        im.save(f"{output_directory}/probes_{session_id}_crop.png")


    elif plot_type == "video":

        anim = Animation(scene, "videos", "probes", size=None)

        anim.add_keyframe(0, camera=cam0, zoom=0.8)
        anim.add_keyframe(2, camera=cam1, zoom=1)
        anim.add_keyframe(4, camera=cam2, zoom=0.8)
        anim.add_keyframe(6, camera=cam3, zoom=1)

        # Make videos
        anim.make_video(duration=8, fps=5)
