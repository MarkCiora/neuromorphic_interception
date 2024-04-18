from neuro_cam import NVS
from object import obj
import numpy as np
import cv2
from origLP import origLP

pos = np.asarray([-50, 0, 0], dtype=np.float32)
vel = np.asarray([5, 0, 0], dtype=np.float32)
up = np.asarray([0, 1, 1], dtype=np.float32)
cam = NVS(250,1,1,pos,vel,up)
target = obj(np.asarray([0, -15,-5], dtype=np.float32),.3)

optical_flow = origLP(cam.res, cam.res, cam.dt)

ac = np.zeros((1,3), dtype = np.float32)

video_writer = cv2.VideoWriter('output_video.mp4', 
                               cv2.VideoWriter_fourcc(*'mp4v'), 
                               30, 
                               (1500, 500))

while True:
    cam.update_physics(ac)
    tframe,sframe,frame = cam.update_frame(target)

    flow_events = optical_flow.process_frame(tframe)

    avg_vx = 0
    avg_vy = 0
    center_x = 0
    center_y = 0
    count = 0
    for flow_event in flow_events:
        avg_vx = avg_vx + flow_event.vx
        avg_vy = avg_vy + flow_event.vy
        center_x = center_x + flow_event.tc.x
        center_y = center_y + flow_event.tc.y
        count = count + 1

    if count > 0:
        avg_vx = avg_vx / count
        avg_vy = avg_vy / count
        center_x = center_x / count
        center_y = center_y / count
    else:
        avg_vx = 0
        avg_vy = count
        center_x = count
        center_y = count

    ac[0,2] = np.sqrt(avg_vx*avg_vx + avg_vy*avg_vy)
    if ac[0,2] < 0.00001:
        ac[0,0] = 0
        ac[0,1] = 0
    else:
        ac[0,0] = avg_vx / ac[0,2]
        ac[0,1] = avg_vy / ac[0,2]
    ac[0,2] /= 100

    print("avg_vx = ", avg_vx)
    print("avg_vy = ", avg_vy)
    print("center_x = ", center_x)
    print("center_y = ", center_y)
    print("ac: ", ac)

    LOS, dLOS = cam.estimate_LOS(tframe, sframe)
    # ac = cam.calc_ac(LOS, dLOS)

    #get frame pos from LOS
    center = (int((cam.res-1)/2), int((cam.res-1)/2))
    index = LOS * cam.res / cam.fov + cam.res/2
    LOS_end = (round(index[0,1]), round(index[0,0]))
    index += dLOS * cam.res / cam.fov
    dLOS_end = (round(index[0,1]), round(index[0,0]))

    sframe = (sframe + 1).astype(np.uint8) * 127
    tframe = (tframe + 1).astype(np.uint8) * 127
    image = cv2.hconcat([(frame*255).astype(np.uint8), tframe, sframe])
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.arrowedLine(image, center, LOS_end, (0,255,0), 1)
    image = cv2.arrowedLine(image, LOS_end, dLOS_end, (0,0,255), 1)

    image = cv2.resize(image, (1500, 500))

    video_writer.write(image)
    # cv2.imshow("image --- temporal difference --- spatial difference", image)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

    if np.linalg.norm(cam.pos - target.pos) <= target.radius:
        print("hit: ", np.linalg.norm(cam.pos - target.pos))
        break

    if np.dot(cam.vel, target.pos - cam.pos) <= -0:
        print("miss: ", np.linalg.norm(cam.pos - target.pos))
        break
    
    print(np.linalg.norm(cam.pos - target.pos))

video_writer.release()

print("done")