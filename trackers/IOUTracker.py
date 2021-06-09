from trackers.BaseTracker import BaseTracker


class IOUTracker(BaseTracker):
    def __init__(self):
        super(BaseTracker, self).__init__()
        self.tracker = {}
        self.frame_num = 0
        self.maxed_id = 0

    def update(self, preds):
        is_Track = {}  # 用来标志上一帧的目标是否被跟踪上

        T_IOU = 0.1  # 判断IOU的阈值

        # 当前帧是第一帧，初始化跟踪字典
        if self.frame_num == 0:
            count = 0  # 计数,id值
            for pred in preds:  # 当前检测到的目标加入字典
                rec = pred[:4].tolist()
                self.tracker[count] = rec
                if self.maxed_id < count:
                    self.maxed_id = count
                count += 1  # 计数+1

        # 初始化is_Track,标志是否处理过
        for track in list(self.tracker.keys()):
            is_Track[track] = 0

        # 首先利用阈值对检测器的检测结果进行过滤，是否能返回检测置信度?

        # 非初始帧，追踪位置，设置ID号
        if(self.frame_num != 0):
            # 对当前帧每一个检测位置，找与其IOU最大的跟踪位置
            for pred in preds:
                rec1 = pred[:4].tolist()
                # rec1 = []  # 当前帧检测框的矩形
                # 遍历所有的跟踪框,找最大IOU
                max_IOU = -1
                max_id = 0  # 记录最大IOU的跟踪框的ID
                for id, track in zip(self.tracker.keys(), list(self.tracker.values())):
                    rec2 = []
                    rec2.append(track[0])
                    rec2.append(track[1])
                    rec2.append(track[2])
                    rec2.append(track[3])
                    current_IOU = self.IOU(rec1, rec2)
                    if(max_IOU < 0):  # 初始化最大值
                        max_IOU = current_IOU
                        max_id = id
                    if(current_IOU > max_IOU):  # 找最大值
                        max_IOU = current_IOU
                        max_id = id
                # print(max_IOU)

                # 判断IOU是否大于给定的阈值
                if(max_IOU > T_IOU):  # 大于，说明是一个跟踪中的目标，更新跟踪位置
                    self.tracker[max_id] = rec1
                    is_Track[max_id] = 1
                elif(0 < max_IOU <= T_IOU):  # 小于阈值，进一步判断
                    # 判断轨迹已经持续的帧数是否大于阈值?大于阈值则认为目标消失了?
                    # 认为目标消失,从字典中剔除
                    self.tracker.pop(max_id, None)
                    is_Track.pop(max_id, None)

                # 没有相匹配的框,认为出现了一个新目标，目标加入字典,id值取最大id值+1
                if(max_IOU <= 0):
                    if self.tracker:
                        # temp=int(list(Tracker.keys())[-1])+1
                        self.maxed_id += 1  # 更新最大值
                        self.tracker[self.maxed_id] = rec1
                    else:
                        self.tracker[0] = rec1

            # 看看之前帧跟踪的目标是否都已操作过，若未操作则剔除
            for id, is_track in zip(is_Track.keys(), list(is_Track.values())):
                if(is_track == 0):  # 剔除目标
                    self.tracker.pop(id, None)

        return zip(self.tracker.keys(), list(self.tracker.values()))

    def IOU(self, rec1, rec2):
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
