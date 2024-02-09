import cv2
from object import obj
import numpy as np

class NVS:

    #fov in radians
    #sens will be how sensitive to change
    spatial_filter = np.asarray([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype = np.float32)
    threshold = .3

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
        pd_dir = pd / np.linalg.norm(pd)
        view_width = 2 * np.sin(self.fov * 0.5) * np.linalg.norm(pd)
        pixel_rad = view_width * 0.5 / self.res
        print(pd, self.vel)

        new_frame = np.zeros((self.res,self.res), dtype=np.float32)
        for i in range(self.res):
            for j in range(self.res):
                #check if pixel i,j sees it
                theta = i*self.fov/(self.res-1) - self.fov/2
                phi = j*self.fov/(self.res-1) - self.fov/2
                dir = self.vel + self.right * theta + self.up * phi
                dir = dir / np.linalg.norm(dir)
                r = np.arccos(np.dot(pd_dir, dir))
                r = np.linalg.norm(pd) * np.arctan(r)
                if r <= object.radius - pixel_rad:
                    new_frame[i,j] = 1.0
                elif r <= object.radius + pixel_rad:
                    lol = (r - object.radius - pixel_rad) * 1.5708 / pixel_rad
                    new_frame[i,j] = (1.0 - np.cos(lol)) * 0.5
                else:
                    new_frame[i,j] = 0.0
        
        tframe = new_frame
        sframe = cv2.filter2D(new_frame, -1, NVS.spatial_filter) * 0.5 + 0.5
        sframe = (sframe > 0.5 + NVS.threshold).astype(np.int8) - \
                 (sframe < 0.5 - NVS.threshold).astype(np.int8)
        sframe = sframe * 127
        return tframe, sframe