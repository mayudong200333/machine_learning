import pybullet as p
import time
import pybullet as p
import time
import pybullet_data
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from base64 import b64encode
from IPython.display import HTML
import numpy as np

def save_video(frames, path):
    clip = mpy.ImageSequenceClip(frames, fps=30)
    clip.write_videofile(path, fps=30)

def play_mp4(path):
    mp4 = open(path, 'rb').read()
    url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""<video width=400 controls><source src="%s" type="video/mp4"></video>""" % url)

# 一回だけ実行してください
physicsClient = p.connect(p.GUI)  # ローカルで実行するときは、p.GUI を指定してください

# 床を出現させます
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
timestep = 1. / 240.
p.setTimeStep(timestep)
planeId = p.loadURDF("plane.urdf")

