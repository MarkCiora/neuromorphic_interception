from neuro_cam import NVS
from object import obj
import numpy as np
import cv2

pos = np.asarray([-50, 0, 0], dtype=np.float32)
vel = np.asarray([5, 0, 0], dtype=np.float32)
up = np.asarray([0, 1, 1], dtype=np.float32)
cam = NVS(100,1,1,pos,vel,up)
target = obj(np.asarray([0, -15,-5], dtype=np.float32),.3)

ac = np.zeros((1,3), dtype = np.float32)
while True:
    cam.update_physics(ac)
    tframe,sframe = cam.update_frame(target)
    LOS, dLOS = cam.estimate_LOS(tframe, sframe)
    ac = cam.calc_ac(LOS, dLOS)

    #get frame pos from LOS
    center = (int((cam.res-1)/2), int((cam.res-1)/2))
    index = LOS * cam.res / cam.fov + cam.res/2
    LOS_end = (round(index[0,1]), round(index[0,0]))
    index += dLOS * cam.res / cam.fov
    dLOS_end = (round(index[0,1]), round(index[0,0]))

    image = (sframe + 1).astype(np.uint8) * 127
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.arrowedLine(image, center, LOS_end, (0,255,0), 1)
    image = cv2.arrowedLine(image, LOS_end, dLOS_end, (0,0,255), 1)

    cv2.imshow("ok", image)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

    if np.linalg.norm(cam.pos - target.pos) <= target.radius:
        print("hit: ", np.linalg.norm(cam.pos - target.pos))
        break

    if np.dot(cam.vel, target.pos - cam.pos) <= -0:
        print("miss: ", np.linalg.norm(cam.pos - target.pos))
        break