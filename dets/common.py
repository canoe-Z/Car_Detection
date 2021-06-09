from cv2 import cv2
import numpy as np
import imutils

import os.path as osp
import os


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def readtxt(label_file):
    with open(label_file) as f:
        for line in f.readlines():
            line = line.strip('\n')
            value = line.split(' ')
            label = value[0]
            bx = eval(value[1])
            by = eval(value[2])
            bw = eval(value[3])
            bh = eval(value[4])
    return label, bx, by, bw, bh


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def convert_bouding_box(img, bx, by, bw, bh):
    rows = img.shape[0]
    cols = img.shape[1]

    xmin = int((bx-bw/2)*cols)
    ymin = int((by-bh/2)*rows)
    xmax = int((bx+bw/2)*cols)
    ymax = int((by+bh/2)*rows)

    return xmin, ymin, xmax, ymax


def convert_yolo_box(img, xmin, ymin, xmax, ymax):
    rows = img.shape[0]
    cols = img.shape[1]

    bx = (xmin+xmax)/2/cols
    by = (ymin+ymax)/2/rows
    bw = abs((xmax-xmin)/cols)
    bh = abs((ymax-ymin)/rows)

    return bx, by, bw, bh


def draw_bouding_box_yolo(img, bx, by, bw, bh, id: int = None, conf=None):
    xmin, ymin, xmax, ymax = convert_bouding_box(img, bx, by, bw, bh)
    draw_bouding_box(img, xmin, ymin, xmax, ymax, id, conf)


def draw_bouding_box(img, xmin, ymin, xmax, ymax, id: int = None, conf=None):
    x1 = int(xmin)
    y1 = int(ymin)
    x2 = int(xmax)
    y2 = int(ymax)

    if conf == None:
        plot_one_box((x1, y1, x2, y2), img, (0, 0, 255), str(id), 3)
    else:
        plot_one_box((x1, y1, x2, y2), img, (0, 0, 255),
                     str(id)+'{:.2f}'.format(conf), 3)


# 滑窗
def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


# 金字塔
def pyramid(image, scale=1.2, minSize=(30, 30)):
    yield image

    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        yield image

# From YOLO


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    tl = line_thickness or round(
        0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
