from dets.HaarDectector import HaarDetector
from cv2 import cv2
from dets.YOLODetector import YOLODetector
from dets.common import draw_bouding_box_yolo, draw_bouding_box
import time


def main():
    det = YOLODetector('./models/yolo/best_bdd100.pt', 0.25)
    #det = HaarDetector()
    cap = cv2.VideoCapture('./data/video-02.mp4')
    i=0
    Tracker={}
    while True:
        start_time = time.time()

        _, frame = cap.read()
        if frame is None:
            break
        preds = det.detect(frame)

        for pred in preds:
            bx, by, bw, bh, conf, _ = pred
            draw_bouding_box(frame, bx, by, bw, bh, conf)
            #draw_bouding_box_yolo(frame, bx, by, bw, bh, conf)
        end_time = time.time()
        if (end_time != start_time):
            fps = 1.0/(end_time - start_time)
        i+=1
        cv2.putText(frame, 'FPS: ' + str(int(fps)), (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        print(fps)
        cv2.imshow('test', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
