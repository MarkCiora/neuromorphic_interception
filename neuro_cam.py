import cv2
from object import obj
import numpy as np

class NVS:

    #fov in radians
    #sens will be how sensitive to change

    def __init__(self, res, fov, sens, pos, vel, up):
        self.res = res
        self.fov = fov
        self.sens = sens
        self.pos = pos
        self.vel = vel
        self.up = up
        self.right = np.cross(self.vel, self.up)

    def update_frame(self, object:obj):
        pd = object.pos - self.pos

        for i in range(self.res):
            for j in range(self.res):
                #check if pixel i,j sees it
                theta = i*self.fov/(self.res-1) - self.fov/2
                phi = j*self.fov/(self.res-1) - self.fov/2
                dir = self.vel + self.right * theta + self.up * phi
                dir = dir / np.linalg.norm(dir)
                print(dir)