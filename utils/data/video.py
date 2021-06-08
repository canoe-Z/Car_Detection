import cv2
import numpy as np
import os
 
name_list=[]

def getdir(dirpath,level = 0):
    level += 1
    if not dirpath:
        dirpath = os.getcwd()

    mylist= os.listdir(dirpath)
    for name in mylist:
        name = dirpath + '\\\\' + name
        if os.path.isdir(name):
            getdir(name,level)
        else:
            name_list.append(name)

if __name__ == "__main__":
    img_root = './MVI_39781'  # 是图片序列的位置
    fps = 50  # 可以随意调整视频的帧速率
    getdir(img_root)
    #可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter('TestVideo.avi',fourcc,fps,(960,540),True)#最后一个是保存图片的尺寸

    for name in name_list:
        #print(name)
        frame = cv2.imread(name)
        # cv2.imshow('frame',frame)
        videoWriter.write(frame)
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #         break
    videoWriter.release()
    cv2.destroyAllWindows()

