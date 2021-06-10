from cv2 import cv2
from dets.YOLODetector import YOLODetector
from dets.common import draw_bouding_box
from trackers.IOUTracker import IOU_Tracker
from trackers.sort import * 


def main():
    det = YOLODetector('./models/yolo/best_ua.pt', 0.4)
    tracker = IOU_Tracker()
    mot_tracker = Sort()#Sort方法多目标追踪 
    cap = cv2.VideoCapture('./data/TestVideo.avi')
    cv2.namedWindow('test')
    for _ in range(1000): 
        _, frame = cap.read()
        if frame is None: 
            break
        preds = det.detect(frame)#preds是numpy array 一行代表一个目标
        #IOU Track方法
        tracker.init(preds,frame)
        tracker.update()
        tracker.draw_bounding()
        tracker.show()
        
        '''
        #Sort方法
        detections=[]#传入sort.update() [x1,y1,x2,y2,score]
        for pred in preds.tolist():
            temp=[]
            temp.append(pred[0])
            temp.append(pred[1])
            temp.append(pred[2])
            temp.append(pred[3])
            temp.append(pred[4])
            detections.append(temp)
        track_bbs_ids = mot_tracker.update(np.array(detections))#track_bbs_ids  np.array()每行包括box和id
        track_list=track_bbs_ids.tolist()#np.array转为list
        for track in track_list:
            bx, by, bw, bh=track[0],track[1],track[2],track[3]
            draw_bouding_box(frame, bx, by, bw, bh, track[4])
        cv2.imshow('test', frame)
        cv2.waitKey(33)
        '''
        '''
        for id, track in tracks:
            x1, y1, x2, y2 = track
            draw_bouding_box(frame, x1, y1, x2, y2, id)
        cv2.imshow('test', frame)
        cv2.waitKey(0)
        '''


if __name__ == "__main__":
    main()
