from dets.BaseDetector import baseDet
from cv2 import cv2
import numpy as np


class HaarDetector(baseDet):
    def __init__(self):
        super(HaarDetector, self).__init__()
        self.init_model()

    def init_model(self):
        self.carCascade = cv2.CascadeClassifier(
            './models/haar.xml')  # 调用haar级联分类器

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def detect(self, img):
        gray = self.preprocess(img)
        cars = self.carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
        pred_boxes = []
        for (x1, y1, w, h) in cars:
            x2 = x1+w
            y2 = y1+h
            pred_boxes.append([int(x1), int(y1), int(x2), int(y2), 1, 0])
        return np.array(pred_boxes)
