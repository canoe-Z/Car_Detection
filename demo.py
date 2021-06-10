from cv2 import cv2
from dets.YOLODetector import YOLODetector
from dets.common import draw_bbs_ids
from trackers.IOUTracker import IOUTracker
from trackers.sort import *
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch


def main():
    cfg = get_config()
    cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    det = YOLODetector('./models/yolo/best_300.pt', 0.5)
    tracker = IOUTracker()
    # tracker = Sort()  # Sort方法多目标追踪
    cap = cv2.VideoCapture('./data/video-02.mp4')
    cv2.namedWindow('test')

    #result = []
    for _ in range(1000):
        _, frame = cap.read()
        if frame is None:
            break
        dets = det.detect(frame)  # preds是numpy array 一行代表一个目标
        # IOU Track方法
        outputs = tracker.update(dets)
        #print(tf)

        # bbox_xywh = []
        # confs = []
        # clss = []

        # # Adapt detections to deep sort input format
        # if len(dets):
        #     for x1, y1, x2, y2, conf, _ in dets:

        #         obj = [
        #             int((x1+x2)/2), int((y1+y2)/2),
        #             int(x2-x1), int(y2-y1)
        #         ]
        #         bbox_xywh.append(obj)
        #         confs.append(conf)
        #         clss.append(1)

        #     xywhs = torch.Tensor(bbox_xywh)
        #     confss = torch.Tensor(confs)

        #     # Pass detections to deepsort
        #     outputs = deepsort.update(xywhs, confss, frame)
        #     print(outputs)

        draw_bbs_ids(frame, outputs)

        # Sort方法
        # track = mot_tracker.update(dets)
        # draw_bbs_ids(frame, track)`

        cv2.imshow('test', frame)
        cv2.waitKey(1)

    print('Final')
    # output_video('./out/result.avi', result)
    '''
        for id, track in tracks:
            x1, y1, x2, y2 = track
            draw_bouding_box(frame, x1, y1, x2, y2, id)
        cv2.imshow('test', frame)
        cv2.waitKey(0)
        '''


if __name__ == "__main__":
    main()
