from neuro_cam import NVS
from object import obj
import numpy as np

pos = np.asarray([-10, 0, 0], dtype=np.float32)
vel = np.asarray([1, 0, 0], dtype=np.float32)
up = np.asarray([0, 0, 1], dtype=np.float32)
cam = NVS(4,0.5,1,pos,vel,up)
target = obj(np.asarray([0, 0, 0], dtype=np.float32),1)

cam.update_frame(target)