from cv2 import cv2
from dets.YOLODetector import YOLODetector
from dets.common import draw_bouding_box
from trackers.IOUTracker import IOUTracker


def main():
    det = YOLODetector('./models/yolo/best_ua.pt', 0.4)
    tracker = IOUTracker()
    cap = cv2.VideoCapture('./data/TestVideo.avi')
    cv2.namedWindow('test')
    for _ in range(1000):
        _, frame = cap.read()
        if frame is None:
            break
        preds = det.detect(frame)
        tracks = tracker.update(preds)

        for id, track in tracks:
            x1, y1, x2, y2 = track
            draw_bouding_box(frame, x1, y1, x2, y2, id)

        cv2.imshow('test', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
