'''
code by CANOE@2021/06/04
Reference https://blog.csdn.net/weixin_40769885/article/details/82869760
'''

import random
import os
import shutil
import glob

import os.path

from dets.common import mkdirs


def moveFile(fileDir, tarDir, rate=0.1):
    mkdirs(tarDir)
    pwd = os.getcwd()
    os.chdir(fileDir)
    pathDir = glob.glob('*.jpg')  # 取图片的原始路径
    os.chdir(pwd)
    filenumber = len(pathDir)
    # rate = 0.25  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber*rate)  # 按照rate比例从文件夹中取一定数量图片
    samples = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    for sample in samples:
        name = sample.split(".")[0]
        shutil.move(fileDir+'/'+name+'.jpg', tarDir+'/'+name+'.jpg')
        shutil.move(fileDir+'/'+name+'.txt', tarDir+'/'+name+'.txt')
    print("Move Finished!")
    return


def delete_file(fileDir):
    pwd = os.getcwd()
    os.chdir(fileDir)
    pathDir = glob.glob('*.jpg')  # 取图片的原始路径
    os.chdir(pwd)
    for sample in pathDir:
        name = sample.split(".")[0]
        if os.path.isfile(fileDir+'/'+name+'.txt') == False:
            os.remove(fileDir+'/'+name+'.jpg')
    print("Delete Finished!")
    return


def main():
    test_path = "../DETRAC-dataset-yolo/test"
    train_path = "../DETRAC-dataset-yolo/train"
    val_path = "../DETRAC-dataset-yolo/val"

    delete_file(train_path)
    delete_file(test_path)
    moveFile(train_path, val_path)


if __name__ == "__main__":
    main()
