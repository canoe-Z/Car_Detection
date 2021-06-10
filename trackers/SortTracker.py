from cv2 import cv2
from trackers.sort import *
#IOU track类
class SortTracker(): 
    #追踪器
    def update(dets):
        detections = dets[:5]
        return mot_tracker.update(detections)
        track_list = track_bbs_ids.tolist()  # np.array转为list
        for track in track_list:
            bx, by, bw, bh = track[0], track[1], track[2], track[3]
            draw_bouding_box(frame, bx, by, bw, bh, track[4])
