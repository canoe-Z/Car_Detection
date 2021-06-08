from cv2 import cv2
from utils.HaarDectector import HaarDetector
from utils.common import draw_bouding_box_yolo


def main():
    haar = HaarDetector()

    cap = cv2.VideoCapture('./data/cars.mp4')
    cv2.namedWindow('test')
    for _ in range(100):
        _, frame = cap.read()
        if frame is None:
            break
        preds = haar.detect(frame)
        for pred in preds:
            bx, by, bw, bh = pred
            draw_bouding_box_yolo(frame, bx, by, bw, bh)
        cv2.imshow('test', frame)
        cv2.waitKey(20)


if __name__ == "__main__":
    main()
