from brainrender import Scene
from brainrender.video import VideoMaker


from rich import print
from myterial import orange
from pathlib import Path

# Create a scene
scene = Scene("my video")
scene.add_brain_region("TH")

# Create an instance of video maker
vm = VideoMaker(scene, "./examples", "vid1")

# make a video with the custom make frame function
# this just rotates the scene
vm.make_video(azimuth=2, duration=5, fps=15)

