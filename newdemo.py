from cv2 import cv2
from dets.YOLODetector import YOLODetector
from dets.common import draw_bouding_box, draw_bbs_ids
from trackers.IOUTracker import IOU_Tracker
from trackers.sort import *


def main():
    det = YOLODetector('./models/yolo/best_300.pt', 0.5)
    tracker = IOU_Tracker()
    mot_tracker = Sort()  # Sort方法多目标追踪
    cap = cv2.VideoCapture('./data/TestVideo.avi')
    cv2.namedWindow('test')
    for _ in range(1000):
        _, frame = cap.read()
        if frame is None:
            break
        dets = det.detect(frame)  # preds是numpy array 一行代表一个目标
        # IOU Track方法
        # tracker.init(preds,frame)
        # tracker.update()
        # tracker.draw_bounding()
        # tracker.show()

        # Sort方法
        track = mot_tracker.update(dets)
        draw_bbs_ids(frame, track)
        cv2.imshow('test', frame)
        cv2.waitKey(1)

        '''
        for id, track in tracks:
            x1, y1, x2, y2 = track
            draw_bouding_box(frame, x1, y1, x2, y2, id)
        cv2.imshow('test', frame)
        cv2.waitKey(0)
        '''


if __name__ == "__main__":
    main()
