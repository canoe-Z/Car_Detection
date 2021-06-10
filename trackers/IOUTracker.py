from cv2 import cv2
from dets.common import draw_bouding_box
#IOU track类
class IOU_Tracker(): 
    def __init__(self):
        self.frames=0
        self.Tracker={}#全局
        self.maxed_id=0
    #每帧初始化预测值及图像
    def init(self,detect,pic):
        self.preds=detect.tolist()
        self.img=pic
    #追踪器
    def update(self):
        T_IOU=0.5#判断IOU的阈值
        is_Track={}#用来标志上一帧的目标是否被跟踪上
        rows = self.img.shape[0]
        cols = self.img.shape[1]
        #当前帧是第一帧，初始化跟踪字典
        if(self.frames==0):
            for count,pred in enumerate(self.preds):#当前检测到的目标加入字典
                rec=[]
                rec.append(pred[0])
                rec.append(pred[1])
                rec.append(pred[2])
                rec.append(pred[3])
                self.Tracker[count]=rec
                if(self.maxed_id<count):
                    self.maxed_id=count
        #初始化is_Track,标志是否处理过            
        for track in list(self.Tracker.keys()):
            is_Track[track]=0

        #首先利用阈值对检测器的检测结果进行过滤，是否能返回检测置信度?

        #非初始帧，追踪位置，设置ID号
        if(self.frames!=0):
            #对当前帧每一个检测位置，找与其IOU最大的跟踪位置     
            for pred in self.preds:
                rec1=[]#当前帧检测框的矩形
                rec1.append(pred[0])
                rec1.append(pred[1])
                rec1.append(pred[2])
                rec1.append(pred[3])
                #遍历所有的跟踪框,找最大IOU
                max_IOU=-1
                max_id=0#记录最大IOU的跟踪框的ID
                for id,track in zip(self.Tracker.keys(),list(self.Tracker.values())):
                    rec2=[]
                    rec2.append(track[0])
                    rec2.append(track[1])
                    rec2.append(track[2])
                    rec2.append(track[3])
                    current_IOU=self.IOU(rec1,rec2)####
                    if(max_IOU<0):#初始化最大值
                        max_IOU=current_IOU
                        max_id=id
                    if(current_IOU>max_IOU):#找最大值
                        max_IOU=current_IOU
                        max_id=id
                #判断IOU是否大于给定的阈值    
                if(max_IOU>T_IOU):#大于，说明是一个跟踪中的目标，更新跟踪位置
                    self.Tracker[max_id]=rec1
                    is_Track[max_id]=1
                elif(0<max_IOU<=T_IOU):#小于阈值，进一步判断
                    #判断轨迹已经持续的帧数是否大于阈值?大于阈值则认为目标消失了?
                    #认为目标消失,从字典中剔除
                    self.Tracker.pop(max_id,None)
                    is_Track.pop(max_id,None)
                #没有相匹配的框,认为出现了一个新目标，目标加入字典,id值取最大id值+1
                if(max_IOU<=0):
                    self.maxed_id+=1#更新最大值
                    self.Tracker[self.maxed_id]=rec1
            #看看之前帧跟踪的目标是否都已操作过，若未操作则剔除
            for id,is_track in zip(is_Track.keys(),list(is_Track.values())):
                if(is_track==0):#剔除目标
                    self.Tracker.pop(id,None)
        self.frames+=1
    #计算两个矩形框的IOU,矩形框用左上角坐标和右下角坐标表示
    def IOU(self,rec1,rec2):
        summ=((rec1[3]-rec1[1])*(rec1[2]-rec1[0]))+((rec2[3]-rec2[1])*(rec2[2]-rec2[0]))
        left=max(rec1[0],rec2[0])
        right=min(rec1[2],rec2[2])
        top=max(rec1[1],rec2[1])
        bottom=min(rec1[3],rec2[3])
        if left>=right or top>=bottom:
            return 0
        else:
            inter=(right-left)*(bottom-top)
            return (inter/(summ-inter))*1.0 
    #画出跟踪框
    def draw_bounding(self):
        for id,track in zip(self.Tracker.keys(),list(self.Tracker.values())):
            bx, by, bw, bh = track
            draw_bouding_box(self.img, bx, by, bw, bh, id)
    #显示
    def show(self):
        cv2.imshow('test',self.img)
        cv2.waitKey(1)
