import random
import numpy as np
import cv2

import common
import glob
import os
from tqdm import tqdm


class pick(object):
    def __init__(self, img, box, count, width=128, height=128):
        '''
        box是矩阵，存对应[第几个目标][起点x 起点y width height]
        '''
        self.img = img
        self.w = img.shape[1]
        self.h = img.shape[0]
        self.height = height
        self.width = width
        self.number = len(box)
        self.box = box
        self.count = count

    def pickneg(self):
        x = random.randint(0, self.w-self.width)
        y = random.randint(0, self.h-self.height)
        return x, y

    def save_pic(self, x, y):
        img = self.img
        common.mkdirs('neg_pick')
        neg_path = "neg_pick\\"+str(self.count)+'.jpg'
        neg_img = img[y:y+self.height, x:x+self.width]
        cv2.imwrite(neg_path, neg_img)

    # 两个检测框框是否有交叉，如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
    def bb_overlab(self):
        '''
        说明：图像中，从左往右是 x 轴（0~无穷大），从上往下是 y 轴（0~无穷大），从左往右是宽度 w ，从上往下是高度 h
        :param x1: 第一个框的左上角 x 坐标
        :param y1: 第一个框的左上角 y 坐标
        :param w1: 第一幅图中的检测框的宽度
        :param h1: 第一幅图中的检测框的高度
        :param x2: 第二个框的左上角 x 坐标
        :param y2:
        :param w2:
        :param h2:
        :return: 两个如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
        '''
        box = self.box
        num = self.number
        while(1):
            result = []
            x2, y2 = self.pickneg()
            w2 = self.width
            h2 = self.height
            for i in range(num):
                x1 = self.box[i][0]
                y1 = self.box[i][1]
                w1 = self.box[i][2]
                h1 = self.box[i][3]
                if(x1 > x2+w2):
                    result.append(0)
                    continue
                if(y1 > y2+h2):
                    result.append(0)
                    continue
                if(x1+w1 < x2):
                    result.append(0)
                    continue
                if(y1+h1 < y2):
                    result.append(0)
                    continue
                else:
                    result.append(1)
            result = np.array(result)
            if((result == 0).all()):
                self.save_pic(x2, y2)
                return
            #     colInt = abs(min(x1 +w1 ,x2+w2) - max(x1, x2))
            #     rowInt = abs(min(y1 + h1, y2 +h2) - max(y1, y2))
            #     overlap_area = colInt * rowInt
            #     result.append(1)


def main():
    src = 'D:/Code/Car/Dataset/DETRAC-dataset-yolo/train'
    path = glob.glob(os.path.join(src, '*.jpg'))
    pbar = tqdm(path)
    count = 0
    for file in pbar:
        box = []
        img = cv2.imread(file)
        label_filename = os.path.split(file)[1].split('.')[0]+'.txt'
        label_file = os.path.join(src, label_filename)
        label = common.readtxt(label_file)
        bx, by, bw, bh = label[1:]
        xmin, xmax, ymin, ymax = common.convert_bouding_box(
            img, bx, by, bw, bh)
        box.append([xmin, ymin, xmax-xmin, ymax-ymin])
        size = random.randint(64, 320)
        a = pick(img, box, count, size, size)
        a.bb_overlab()
        count += 1


if __name__ == "__main__":
    main()
