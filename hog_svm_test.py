import cv2
import numpy as np
from tqdm import tqdm
from cv2 import cv2
from py_cpu_nms import py_cpu_nms
from common import pyramid
from common import sliding_window
from common import draw_bouding_box


def main():
    frames = []
    cap = cv2.VideoCapture('./TestVideo.avi')
    for _ in range(100):
        ret, frame = cap.read()
        if frame is None:
            break
        frames.append(frame)
    #imgs = ['data/0_14.bmp', 'data/0_15.bmp', 'data/0_20.bmp', 'data/1_04.bmp']
    #svm = cv2.ml.SVM_load("model/svm/svm_linear.xml")
    svm = cv2.ml.SVM_load("svm.xml")
    hog = cv2.HOGDescriptor((80, 80), (40, 40), (8, 8), (8, 8), 9)
    num = 0

    for path in tqdm(frames):
        img = path

        rectangles = []
        counter = 0
        w, h = 80, 80

        # 对得到的图进行滑动窗口，取目标区域用于检测(100, 40)为窗口大小，本文应取(80,80)
        for resized in pyramid(img, scale=1.1):  # 对于迭代器迭代出的大小不一的金字塔图片
            scale = float(img.shape[1])/float(resized.shape[1])

            for (x, y, roi) in sliding_window(resized, (80, 80), (10, 10)):
                if roi.shape[1] != w or roi.shape[0] != h:  # 判断是否超纲
                    continue
                test_gradient = hog.compute(roi)

                # svm预测，下一个函数给出预测的置信度，负的越大置信度越高
                _, result = svm.predict(np.array([test_gradient]))
                _, res = svm.predict(
                    np.array([test_gradient]), flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
                score = res[0][0]
                if result[0][0] == 1:
                    if score < -1.7:
                        rx, ry, rx2, ry2 = int(
                            x * scale), int(y * scale), int((x+w) * scale), int((y+h) * scale)
                        rectangles.append([rx, ry, rx2, ry2, -score])
                        counter += 1

        windows = np.array(rectangles)
        keep = py_cpu_nms(windows, 0.2)

        print(len(windows[keep]))

        for (x, y, x2, y2, score) in windows[keep]:
            draw_bouding_box(img, x, y, x2, y2, -score)

        cv2.imwrite('out/svm/'+str(num)+'.jpg', img)
        num += 1


if __name__ == "__main__":
    main()
