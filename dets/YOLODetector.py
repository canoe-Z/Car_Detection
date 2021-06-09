from dets.BaseDetector import baseDet
import torch
from PIL import Image
from cv2 import cv2


class YOLODetector(baseDet):
    def __init__(self, path, conf=0.5):
        super(YOLODetector, self).__init__()
        self.init_model(path, conf)

    def init_model(self, path, conf):
        self.model = torch.hub.load('ultralytics/yolov5',
                                    'custom', path=path)
        self.model.conf = conf

    def preprocess(self, img):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img

    def detect(self, img):
        im = self.preprocess(img)
        results = self.model(im)
        return results.xyxy[0].cpu().numpy()
