from neuro_cam import NVS
from object import obj
import numpy as np
import cv2

pos = np.asarray([-50, 0, 0], dtype=np.float32)
vel = np.asarray([1, 0, 0], dtype=np.float32)
up = np.asarray([0, 0, 1], dtype=np.float32)
cam = NVS(200,0.5,1,pos,vel,up)
target = obj(np.asarray([0, 1, 2], dtype=np.float32),1)

while True:
    tframe,sframe = cam.update_frame(target)
    cv2.imshow("ok", sframe)
    cam.pos[0] += 1

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break