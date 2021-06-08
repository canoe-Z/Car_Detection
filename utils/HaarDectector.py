from utils.BaseDetector import baseDet
from cv2 import cv2

class HaarDetector(baseDet):
    def __init__(self):
        super(HaarDetector, self).__init__()
        self.init_model()

    def init_model(self):
        self.carCascade = cv2.CascadeClassifier('./model/haar.xml')  # 调用haar级联分类器

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def detect(self, img):
        gray = self.preprocess(img)
        cars = self.carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
        rows = img.shape[0]
        cols = img.shape[1]
        pred_boxes = []
        for (xmin, ymin, w, h) in cars:
            bx = (xmin+w/2)/cols
            by = (ymin+h/2)/rows
            bw = w/cols
            bh = h/rows
            pred_boxes.append([bx, by, bw, bh])
        return pred_boxes