from dataclasses import dataclass
import numpy as np

@dataclass
class event_t:
    x: np.uint16    # x position of event
    y: np.uint16    # y position of event
    t: int          # event time in us
    p: int          # polarity of event (1 == on, 0 == off)

@dataclass
class flow_event_t:
    tc: event_t
    vx: float
    vy: float
    mag: float

# Local Planes algorithm
class origLP:
    def __init__(self, frame_height, frame_width, dt):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.dt = dt
        frame_size = frame_height * frame_width
        self.on_event_cloud = np.zeros(frame_size)
        self.off_event_cloud = np.zeros(frame_size)
        self.total_evts = 0
        self.maxDtThreshold = dt*5
        self.time = 0

    def update_time(self):
        self.time += self.dt

    def process_frame(self, tframe):
        flow_events = []

        self.update_event_clouds(tframe)
        
        for i in range(self.frame_height):
            for j in range(self.frame_width):
                if tframe[i,j] != 0:
                    if tframe[i,j] == 1:
                        event = event_t(i,j,self.time,1)
                    elif tframe[i,j] == -1:
                        event = event_t(i,j,self.time,-1)
                    flow_events.append(self.get_flow(event))

        self.update_time()

        return flow_events
    
    def update_event_clouds(self, tframe):
        for i in range(self.frame_height):
            for j in range(self.frame_width):
                if tframe[i,j] != 0:
                    if tframe[i,j] == 1:
                        self.on_event_cloud[i * self.frame_width + j] = self.time
                    elif tframe[i,j] == -1:
                        self.off_event_cloud[i * self.frame_width + j] = self.time
                

    def get_flow(self, event: event_t):
        # add event to event cloud
        # last = 0
        # if event.p == 1:
        #     last = self.on_event_cloud[event.x * self.frame_width + event.y]
        #     self.on_event_cloud[event.x * self.frame_width + event.y] = event.t
        # else:
        #     last = self.off_event_cloud[event.x * self.frame_width + event.y]
        #     self.off_event_cloud[event.x * self.frame_width + event.y] = event.t

        # maybe add check if pixel has been inactive for at least t_refract
        lf = self.compute_local_plane(event, 5)
        self.total_evts += 1
        if lf.mag != 0:
            return lf
        else:
            return flow_event_t(event, 0, 0, 0)

    def compute_local_plane(self, event: event_t, searchDistance: int):
        if event.p == 1:
            event_cloud = self.on_event_cloud
        else:
            event_cloud = self.off_event_cloud
        
        a10 = 0
        a01 = 0
        ii = 0
        jj = 0

        # idk what these variable values should be
        cut = 1
        thr = 1e-9

        # search area along x axis
        for i in range(-searchDistance, searchDistance+1):
            # search area along y axis
            for j in range(-searchDistance, searchDistance+1):
                # check if pixel is in frame
                if (event.x+1 >= 0 and event.x+i < self.frame_height and event.y+j >= 0 and event.y+j < self.frame_width):
                    # get event for current pixel
                    t1 = event_cloud[(event.x + i) * self.frame_width + (event.y + j)]

                    # check if event exists and if event is recent enough
                    if (t1 != 0 and event.t-t1 < self.maxDtThreshold):
                        # search for nearby events along x axis
                        for xx in range(i+1, searchDistance+1):
                            if (event.x+xx >= 0 and event.x+xx < self.frame_height): # check it's in frame
                                t2 = event_cloud[(event.x + xx) * self.frame_width + (event.y + j)]
                                if (t2 != 0 and event.t-t2 < self.maxDtThreshold):
                                    a10 += float(t2-t1)/(xx-i)
                                    ii += 1

                        # search for nearby events along y axis
                        for yy in range(j+1, searchDistance+1):
                            if (event.y+yy >= 0 and event.y+yy < self.frame_width): # check it's in frame
                                t2 = event_cloud[(event.x + i) * self.frame_width + (event.y + yy)]
                                if (t2 != 0 and event.t-t2 < self.maxDtThreshold):
                                    a01 += float(t2-t1)/(yy-j)
                                    jj += 1
        
        if (ii < cut or jj < cut):
            return flow_event_t(event, 0, 0, 0)
        else:
            a10 /= ii
            a01 /= jj

        # a10 *= 1e-6
        # a01 *= 1e-6

        if (abs(a10) < thr and abs(a01) < thr):
            return flow_event_t(event, 0, 0, 0)
        else:
            temp = 1.0 / (a10 * a10 + a01 * a01)
            vx = a10 * temp
            vy = a01 * temp
            mag = round(np.sqrt(vx * vx + vy * vy))
            return flow_event_t(event, vx, vy, mag)