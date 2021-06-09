from cv2 import cv2
from dets.YOLODetector import YOLODetector
from dets.common import draw_bouding_box, convert_yolo_box


maxed_id = 0  # 之前已出现过的最大id


def main():
    det = YOLODetector('./models/yolo/ua_2080ti.pt', 0.5)
    Tracker = {}
    cap = cv2.VideoCapture('./data/cars.mp4')
    cv2.namedWindow('test')
    for frames in range(10000):
        _, frame = cap.read()
        if frame is None:
            break
        preds = det.detect(frame)
        myp = []
        for pred in preds:
            xmin, ymin, xmax, ymax, conf, _ = pred
            bx, by, bw, bh = convert_yolo_box(frame, xmin, ymin, xmax, ymax)
            myp.append([bx, by, bw, bh])

        IOU_Tracker(myp, frames, Tracker, frame)
        # print(Tracker)
        for id, track in zip(Tracker.keys(), list(Tracker.values())):
            bx, by, bw, bh = track
            draw_bouding_box(frame, bx, by, bw, bh, id)
        # plot_rec(frame,Tracker)

        cv2.imshow('test', frame)
        cv2.waitKey(1)

# preds是当前帧预测到的所有位置（列表）,frames是当前帧数,Tracker是之前的跟踪字典,img是当前图像


def IOU_Tracker(preds, frames, Tracker, img):
    global maxed_id
    is_Track = {}  # 用来标志上一帧的目标是否被跟踪上
    T_IOU = 0.1  # 判断IOU的阈值
    rows = img.shape[0]
    cols = img.shape[1]
    # 反归一化,中心坐标以及框的宽高
    for pred in preds:
        pred[0] *= cols
        pred[1] *= rows
        pred[2] *= cols
        pred[3] *= rows
    # 当前帧是第一帧，初始化跟踪字典
    if(frames == 0):
        count = 0  # 计数,id值
        for pred in preds:  # 当前检测到的目标加入字典
            rec = []
            rec.append(pred[0]-pred[2]/2)
            rec.append(pred[1]-pred[3]/2)
            rec.append(pred[0]+pred[2]/2)
            rec.append(pred[1]+pred[3]/2)
            Tracker[count] = rec
            if(maxed_id < count):
                maxed_id = count
            count += 1  # 计数+1
    # 初始化is_Track,标志是否处理过
    for track in list(Tracker.keys()):
        is_Track[track] = 0

    # 首先利用阈值对检测器的检测结果进行过滤，是否能返回检测置信度?

    # 非初始帧，追踪位置，设置ID号
    if(frames != 0):
        # 对当前帧每一个检测位置，找与其IOU最大的跟踪位置
        for pred in preds:
            rec1 = []  # 当前帧检测框的矩形
            rec1.append(pred[0]-pred[2]/2)
            rec1.append(pred[1]-pred[3]/2)
            rec1.append(pred[0]+pred[2]/2)
            rec1.append(pred[1]+pred[3]/2)
            # 遍历所有的跟踪框,找最大IOU
            max_IOU = -1
            max_id = 0  # 记录最大IOU的跟踪框的ID
            for id, track in zip(Tracker.keys(), list(Tracker.values())):
                rec2 = []
                rec2.append(track[0])
                rec2.append(track[1])
                rec2.append(track[2])
                rec2.append(track[3])
                current_IOU = IOU(rec1, rec2)
                if(max_IOU < 0):  # 初始化最大值
                    max_IOU = current_IOU
                    max_id = id
                if(current_IOU > max_IOU):  # 找最大值
                    max_IOU = current_IOU
                    max_id = id
            print(max_IOU)
            # 判断IOU是否大于给定的阈值
            if(max_IOU > T_IOU):  # 大于，说明是一个跟踪中的目标，更新跟踪位置
                Tracker[max_id] = rec1
                is_Track[max_id] = 1
            elif(0 < max_IOU <= T_IOU):  # 小于阈值，进一步判断
                # 判断轨迹已经持续的帧数是否大于阈值?大于阈值则认为目标消失了?
                # 认为目标消失,从字典中剔除
                Tracker.pop(max_id, None)
                is_Track.pop(max_id, None)
            # 没有相匹配的框,认为出现了一个新目标，目标加入字典,id值取最大id值+1
            if(max_IOU <= 0):
                if Tracker:
                    # temp=int(list(Tracker.keys())[-1])+1
                    maxed_id += 1  # 更新最大值
                    Tracker[maxed_id] = rec1
                else:
                    Tracker[0] = rec1
        # 看看之前帧跟踪的目标是否都已操作过，若未操作则剔除
        for id, is_track in zip(is_Track.keys(), list(is_Track.values())):
            if(is_track == 0):  # 剔除目标
                Tracker.pop(id, None)
        # 至此已完成一次循环

# 计算两个矩形框的IOU,矩形框用左上角坐标和右下角坐标表示


def IOU(rec1, rec2):
    summ = ((rec1[3]-rec1[1])*(rec1[2]-rec1[0])) + \
        ((rec2[3]-rec2[1])*(rec2[2]-rec2[0]))
    left = max(rec1[0], rec2[0])
    right = min(rec1[2], rec2[2])
    top = max(rec1[1], rec2[1])
    bottom = min(rec1[3], rec2[3])
    # 判断是否是矩形
    if left >= right or top >= bottom:
        return 0
    else:
        inter = (right-left)*(bottom-top)
        return (inter/(summ-inter))*1.0
# 画出跟踪框


def plot_rec(img, Tracker):
    for id, track in zip(Tracker.keys(), list(Tracker.values())):
        pt1 = (int(track[0]), int(track[1]))
        pt2 = (int(track[2]), int(track[3]))
        cv2.rectangle(img, pt1, pt2, (255, 0, 0))  # 检测框
        cv2.putText(img, str(id), (pt1[0], pt1[1]-2),
                    0, 2, (255, 255, 255), 2, cv2.LINE_AA)  # id值


if __name__ == "__main__":
    main()
