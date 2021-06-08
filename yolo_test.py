import torch
from cv2 import cv2
from PIL import Image


def main():
    model = torch.hub.load('ultralytics/yolov5',
                           'custom', path='./best_ccpd.pt')
    model.conf = 0.5
    cap = cv2.VideoCapture('./test2.flv')
    frames = []
    for _ in range(3000):
        ret, frame = cap.read()
    for _ in range(100):
        ret, frame = cap.read()
        if frame is None:
                break
        frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        frames.append(frame)
    results = model(frames)
    results.save()


if __name__ == "__main__":
    main()
