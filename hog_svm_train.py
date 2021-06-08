import cv2
import numpy as np
from tqdm import tqdm
import os
import glob
import random

from cv2 import cv2

from common import convert_bouding_box


# 提取正样本
def extract_img(images_folder, labels_folder, wsize=(80, 80)):
    imgs = []
    path = glob.glob(os.path.join(images_folder, '*.jpg'))
    pbar = tqdm(path)
    for file in pbar:
        pbar.set_description("Extracting {}:".format(labels_folder))
        img = cv2.imread(file)
        label_filename = os.path.split(file)[1].split('.')[0]+'.txt'
        label_file = os.path.join(labels_folder, label_filename)
        with open(label_file) as f:
            for line in f.readlines():
                line = line.strip('\n')
                value = line.split(' ')
                bx = eval(value[1])
                by = eval(value[2])
                bw = eval(value[3])
                bh = eval(value[4])
                xmin, ymin, xmax, ymax = convert_bouding_box(
                    img, bx, by, bw, bh)
                roi = img[ymin:ymax, xmin:xmax]
                roi_resized = cv2.resize(roi, wsize)
                # cv2.namedWindow('test')
                # cv2.imshow('test',roi_resized)
                # cv2.waitKey(-1)
                imgs.append(roi_resized)

    return imgs


# 获取Hard Sample
def get_neg_hardsample(neg_imgs, svm):
    hard_imgs = []
    hog = cv2.HOGDescriptor((80, 80), (40, 40), (8, 8), (8, 8), 9)
    for neg_img in neg_imgs:
        test_hog = hog.compute(neg_img)
        _, result = svm.predict(np.array([test_hog]))
        if result[0][0] == 1:
            hard_imgs.append(neg_img)
    return hard_imgs


def get_pos_hardsample(pos_imgs, svm):
    hard_imgs = []
    hog = cv2.HOGDescriptor((80, 80), (40, 40), (8, 8), (8, 8), 9)
    for pos_img in pos_imgs:
        test_hog = hog.compute(pos_img)
        _, result = svm.predict(np.array([test_hog]))
        if result[0][0] != 1:
            hard_imgs.append(pos_img)
    return hard_imgs


# 计算HOG特征
def computeHOGs(imgs, wsize=(80, 80)):
    hog_feature = []
    hog = cv2.HOGDescriptor((80, 80), (40, 40), (8, 8), (8, 8), 9)
    for img in tqdm(imgs):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_feature.append(hog.compute(gray))
    return hog_feature


def main():
    pos_imgs = extract_img('../hog_pos/images', '../hog_pos/labels')
    neg_imgs = extract_img('../labelme_neg/images', '../labelme_neg/labels')

    pos_hog = computeHOGs(pos_imgs)
    neg_hog = computeHOGs(neg_imgs)

    labels = []
    for _ in range(len(pos_imgs)):
        labels.append(+1)
    for _ in range(len(neg_imgs)):
        labels.append(-1)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setGamma(0.01)
    svm.setC(10)
    svm.train(np.array(pos_hog+neg_hog), cv2.ml.ROW_SAMPLE, np.array(labels))
    svm.save("svm.xml")

    # 根据初始训练结果获取hard example
    hard_pos_imgs = get_pos_hardsample(pos_imgs, svm)
    hard_pos_hog = computeHOGs(hard_pos_imgs)
    for _ in range(len(hard_pos_imgs)):
        labels.append(1)

    hard_neg_imgs = get_neg_hardsample(neg_imgs, svm)
    hard_neg_hog = computeHOGs(hard_neg_imgs)
    for _ in range(len(hard_neg_imgs)):
        labels.append(-1)

    # 添加hard example后，重新训练
    svm.train(np.array(pos_hog+neg_hog+hard_pos_hog+hard_neg_hog),
              cv2.ml.ROW_SAMPLE, np.array(labels))
    svm.save("svm.xml")


if __name__ == "__main__":
    main()
