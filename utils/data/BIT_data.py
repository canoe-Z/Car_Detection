from math import inf
import numpy as np

import matplotlib.pyplot as plt
import scipy.io as scio

import random as rd
import common

from cv2 import cv2
from tqdm import tqdm
import shutil


def main():
    # 读取数据
    path = '../Dataset/BITVehicle_Dataset/BITVehicle_Dataset'
    dst = '../Dataset/BITVehicle_Dataset/BITVehicle_Dataset_dst'
    common.mkdirs(dst)

    label_mat = scio.loadmat(path+'/'+'VehicleInfo.mat')

    for car_label in tqdm(label_mat['VehicleInfo']):
        img_name = car_label[0][0][0]
        box = car_label[0][3][0][0]
        xmin = box[0][0][0]
        ymin = box[1][0][0]
        xmax = box[2][0][0]
        ymax = box[3][0][0]
        car_type = box[4][0]

        if car_type == 'Sedan' or car_type == 'SUV':
            #bx, by, bw, bh = common.convert_yolo_box(img, xmin, ymin, xmax, ymax)
            radio = (xmax-xmin)/(ymax-ymin)
            if radio > 0.9 and radio < 1.1:
                img = cv2.imread(path+'/'+img_name)
                roi = img[ymin:ymax, xmin:xmax]
                cv2.imwrite(dst+'/'+img_name, roi)


if __name__ == "__main__":
    main()
