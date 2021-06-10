class IOUTracker():
    def __init__(self):
        self.frames = 0
        self.Ta = {}  # {id:[x0,y0,x1,y1,score],...} active
        self.Tf = {}  # finished
        self.last_time = {}  # 记录轨迹持续的时间
        self.highest_score = {}  # 记录轨迹最大的score
        self.maxed_id = 0
        self.set_parameters()
    # 设置跟踪器的参数

    def set_parameters(self):
        self.T_IOU = 0.5
        self.T_Score = 0.5
        self.T_highest = 0.8
        self.T_time = 3
    # 每帧初始化预测值及图像

    # 追踪器
    def update(self, detect):
        self.preds = detect.tolist()  # (1,6)

        is_update = {}
        for tf in self.Tf.keys():
            is_update[tf] = 0
        # 首先利用阈值对检测器的检测结果进行过滤
        for i, pred in enumerate(self.preds):
            if pred[4] < self.T_Score:
                del self.preds[i]
        # 对跟踪字典进行遍历，找检测到的目标中与其IOU最大的目标
        to_del = []
        for id, track in zip(self.Ta.keys(), list(self.Ta.values())):
            rec1 = track[:4]
            max_IOU = -1
            for i, pred in enumerate(self.preds):
                rec2 = pred[:4]
                current_IOU = self.IOU(rec1, rec2)
                if(max_IOU < 0):  # 初始化最大值
                    max_IOU = current_IOU
                    max_pred = pred[:5]
                    max_id = i
                if(current_IOU > max_IOU):  # 找最大值
                    max_IOU = current_IOU
                    max_pred = pred[:5]
                    max_id = i
            # 判断IOU是否大于给定的阈值
            if(max_IOU > self.T_IOU):  # 大于，说明是一个跟踪中的目标，更新跟踪位置
                self.Ta[id] = max_pred
                self.last_time[id] += 1
                if self.highest_score[id] < max_pred[4]:
                    self.highest_score[id] = max_pred[4]
                del self.preds[max_id]
            elif(0 < max_IOU <= self.T_IOU):  # 小于阈值，进一步判断
                self.last_time[id] += 1
                if self.highest_score[id] < max_pred[4]:
                    self.highest_score[id] = max_pred[4]
                if (self.highest_score[id] > self.T_highest) and (self.last_time[id] > self.T_time):
                    self.Tf[id] = max_pred
                    is_update[id] = 1
                    to_del.append(id)
            else:  # 说明未匹配上目标
                to_del.append(id)
        for del_id in to_del:
            self.Ta.pop(del_id, None)
        # 对于检测列表还存在的目标，表明出现了新目标，起始一个新的跟踪
        for pred in self.preds:
            self.maxed_id += 1  # 更新已出现过的最大id
            self.Ta[self.maxed_id] = pred[:5]
            self.last_time[self.maxed_id] = 1
            self.highest_score[self.maxed_id] = pred[4]
        for id, track in zip(self.Ta.keys(), list(self.Ta.values())):
            if (self.highest_score[id] > self.T_highest) and (self.last_time[id] > self.T_time):
                self.Tf[id] = track
                is_update[id] = 1
        for id in is_update.keys():
            if is_update[id] == 0:
                self.Tf.pop(id, None)
        self.frames += 1

        bbs_ids = []
        for id, track in zip(self.Tf.keys(), list(self.Tf.values())):
            x1, y1, x2, y2, score = track
            bbs_ids.append([x1, y1, x2, y2, id])
        return bbs_ids

    # 计算两个矩形框的IOU,矩形框用左上角坐标和右下角坐标表示
    def IOU(self, rec1, rec2):
        summ = ((rec1[3]-rec1[1])*(rec1[2]-rec1[0])) + \
            ((rec2[3]-rec2[1])*(rec2[2]-rec2[0]))
        left = max(rec1[0], rec2[0])
        right = min(rec1[2], rec2[2])
        top = max(rec1[1], rec2[1])
        bottom = min(rec1[3], rec2[3])
        if left >= right or top >= bottom:
            return 0
        else:
            inter = (right-left)*(bottom-top)
            return (inter/(summ-inter))*1.0
