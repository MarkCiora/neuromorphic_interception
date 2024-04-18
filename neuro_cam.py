import cv2
from object import obj
import numpy as np

def normc(a):
    if np.linalg.norm(a) == 0:
        return a
    return a / np.linalg.norm(a)

def remove_component(a,b):
    return normc(normc(a) - normc(b) * np.dot(normc(a), normc(b)))

def project_to_plane(a,b):
    theta = np.arccos(np.dot(normc(a), normc(b)))
    return

class NVS:

    #fov in radians
    #sens will be how sensitive to change
    spatial_filter = np.asarray([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype = np.float32)
    threshold = .001
    dt = 1/5
    N = 3
    acc_max = 1

    def __init__(self, res, fov, sens, pos, vel, up):
        self.res = res
        self.fov = fov
        self.sens = sens
        self.pos = pos
        self.vel = vel
        self.dir = normc(vel)
        self.up = remove_component(up, self.dir)
        self.right = np.cross(self.dir, self.up)
        self.prev_frame = np.zeros((self.res,self.res), dtype=np.float32)

    def calc_ac(self, LOS, dLOS):
        ac = np.zeros((1,3), dtype = np.float32)
        ac[0, 0:2] = dLOS * np.linalg.norm(self.vel) * NVS.N
        ac[0, 2] = 0
        return ac

    def update_physics(self, ac):
        self.up += self.right * NVS.dt * ac[0,2]
        self.up = remove_component(self.up, self.dir)
        self.right = np.cross(self.dir, self.up)

        acceleration = ac[0,1] * self.up + ac[0,0] * self.right
        if np.linalg.norm(acceleration) > NVS.acc_max:
            acceleration = normc(acceleration)
        self.vel += acceleration * NVS.dt
        self.pos += self.vel * NVS.dt

    def update_frame(self, object:obj):
        pd = object.pos - self.pos
        pd_dir = pd / np.linalg.norm(pd)
        view_width = 2 * np.sin(self.fov * 0.5) * np.linalg.norm(pd)
        pixel_rad = view_width * 0.5 / self.res
        #print(pd, self.vel)

        new_frame = np.zeros((self.res,self.res), dtype=np.float32)
        for i in range(self.res):
            for j in range(self.res):
                #check if pixel i,j sees it
                theta = i*self.fov/(self.res-1) - self.fov/2
                phi = j*self.fov/(self.res-1) - self.fov/2
                dir = self.dir + self.right * theta + self.up * phi
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
        
        diff_frame = new_frame - self.prev_frame + 0.5
        tframe = (diff_frame > 0.5 + NVS.threshold).astype(np.int8) - \
                 (diff_frame < 0.5 - NVS.threshold).astype(np.int8)
        tframe = tframe

        sframe = cv2.filter2D(new_frame, -1, NVS.spatial_filter) * 0.5 + 0.5
        sframe = (sframe > 0.5 + NVS.threshold).astype(np.int8) - \
                 (sframe < 0.5 - NVS.threshold).astype(np.int8)
        sframe = sframe

        self.prev_frame = new_frame
        return tframe, sframe, new_frame
    
    def estimate_LOS(self, tframe, sframe):
        mask = np.zeros((self.res,self.res), dtype=np.int8)
        mask_count = 0
        LOS_center = np.zeros((1,2), dtype = np.float32)
        for i in range(self.res):
            for j in range(self.res):
                if sframe[i,j] != 0:
                    for i2 in range(i-2, i+3):
                        for j2 in range(j-2, j+3):
                            if (i2 >=0 and i2 < self.res) and (j2 >=0 and j2 < self.res):
                                mask[i2,j2] = 1
                                
        bcount = 0
        wcount = 0
        bcenter = np.zeros((1,2), dtype = np.float32)
        wcenter = np.zeros((1,2), dtype = np.float32)
        for i in range(self.res):
            for j in range(self.res):
                if mask[i,j] == 1:
                    mask_count += 1
                    if tframe[i,j] == -1:
                        bcount += 1
                    if tframe[i,j] == 1:
                        wcount += 1

        if bcount == 0:
            bcount = 1
            bcenter += self.res * 0.5
        if wcount == 0:
            wcount = 1
            wcenter += self.res * 0.5
        if mask_count == 0:
            mask_count = 1
            LOS_center += self.res * 0.5

        for i in range(self.res):
            for j in range(self.res):
                if mask[i,j] == 1:
                    LOS_center += np.array([i,j],dtype=np.float32) / mask_count
                    if tframe[i,j] == -1:
                        bcenter += np.array([i,j],dtype=np.float32) / bcount
                    if tframe[i,j] == 1:
                        wcenter += np.array([i,j],dtype=np.float32) / wcount

        

        wcenter -= self.res * 0.5
        bcenter -= self.res * 0.5
        LOS_center -= self.res * 0.5
        if bcount == 1:
            bcenter = LOS_center
        if wcount == 1:
            wcenter = LOS_center

        #print(wcenter,bcenter,LOS_center)

        LOS = LOS_center * self.fov / self.res
        

        if wcount + bcount > 2:
            dLOS = normc(wcenter - bcenter) * self.fov / self.res
            dLOS *= (wcount + bcount)
        else:
            dLOS = np.zeros((1,2), dtype = np.float32)

        #print(LOS,dLOS)

        return LOS, dLOS