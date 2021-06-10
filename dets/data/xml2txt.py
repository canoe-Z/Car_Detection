'''
code by CANOE@2021/06/04
code by zzg@2021/05/07
reference to https://www.pythonf.cn/read/111858
'''

import os.path as osp
import os
import xml.dom.minidom as xml
import abc
from tqdm import tqdm


from dets.common import mkdirs


class XmlReader(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    def read_content(self, filename):
        content = None
        if (False == os.path.exists(filename)):
            return content
        filehandle = None
        try:
            filehandle = open(filename, 'rb')
        except FileNotFoundError as e:
            print(e.strerror)
        try:
            content = filehandle.read()
        except IOError as e:
            print(e.strerror)
        if (None != filehandle):
            filehandle.close()
        if(None != content):
            return content.decode("utf-8", "ignore")
        return content

    @abc.abstractmethod
    def load(self, filename):
        pass


class XmlTester(XmlReader):
    def __init__(self):
        XmlReader.__init__(self)

    def load(self, filename):
        filecontent = XmlReader.read_content(self, filename)
        seq_gt = []

        if None != filecontent:
            dom = xml.parseString(filecontent)
            root = dom.getElementsByTagName('sequence')[0]
            if root.hasAttribute("name"):
                seq_name = root.getAttribute("name")
                # print ("*"*20+"sequence: %s" %seq_name +"*"*20)

            # 获取所有的frame
            frames = root.getElementsByTagName('frame')

            for frame in frames:
                if frame.hasAttribute("num"):
                    frame_num = int(frame.getAttribute("num"))
                    # print ("-"*10+"frame_num: %s" %frame_num +"-"*10)

                target_list = frame.getElementsByTagName('target_list')[0]

                # 获取一帧里面所有的target
                targets = target_list.getElementsByTagName('target')
                targets_dic = {}
                for target in targets:
                    if target.hasAttribute("id"):
                        tar_id = int(target.getAttribute("id"))
                        # print ("id: %s" % tar_id)

                    # 获取box
                    box = target.getElementsByTagName('box')[0]
                    if box.hasAttribute("left"):
                        left = box.getAttribute("left")
                        # print ("  left: %s" % left)
                    if box.hasAttribute("top"):
                        top = box.getAttribute("top")
                        # print ("  top: %s" %top )
                    if box.hasAttribute("width"):
                        width = box.getAttribute("width")
                        # print ("  width: %s" % width)
                    if box.hasAttribute("height"):
                        height = box.getAttribute("height")
                        # print ("  height: %s" %height )

                    # 计算中心坐标
                    x = float(left) + float(width)/2
                    y = float(top) + float(height)/2

                    # 获取车辆种类
                    attribute = target.getElementsByTagName('attribute')[0]
                    if attribute.hasAttribute("vehicle_type"):
                        type = attribute.getAttribute("vehicle_type")
                        if type == "car":
                            type = 0
                        if type == "van":
                            type = 1
                        if type == "bus":
                            type = 2
                        if type == "others":
                            type = 3

                    seq_gt.append([frame_num, tar_id, x, y,
                                  float(width), float(height), type])
        return seq_gt


def xml2txt(seq_root, xml_root, label_root):

    seqs = [s for s in os.listdir(seq_root)]

    tid_curr = 0
    tid_last = -1  # 用于在下一个视频序列时，ID数接着上一个视频序列最大值
    for seq in tqdm(seqs):  # 每一个视频序列
        seq_width = 960
        seq_height = 540

        gt_xml = osp.join(xml_root, seq + '.xml')
        reader = XmlTester()
        gt = reader.load(gt_xml)

        # 统计这个序列所有ID
        ids = []
        for line in gt:
            if not line[1] in ids:
                ids.append(line[1])
        # print(ids)

        # 根据ID将同一ID的不同帧标注放在一起
        final_gt = []
        for id in ids:
            for line in gt:
                if line[1] == id:
                    final_gt.append(line)

        # print(len(final_gt))
        # print(final_gt)
        seq_label_root = osp.join(label_root, seq)
        # seq_label_root = label_root

        if not os.path.exists(seq_label_root):
            mkdirs(seq_label_root)

        for fid, tid, x, y, w, h, label in final_gt:
            label = int(label)
            # print(" ",fid,label)
            fid = int(fid)
            tid = int(tid)
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid

            label_fpath = osp.join(seq_label_root, 'img{:05d}.txt'.format(fid))
            # label_fpath = osp.join(seq_label_root, 'img{:05d}_{}.txt'.format(fid, str(seq)))
            label_str = '{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(0,
                                                                    float(x) / seq_width, float(y) / seq_height, float(w) / seq_width, float(h) / seq_height)  # 宽高中心坐标归一化
            with open(label_fpath, 'a') as f:
                f.write(label_str)


def main():
    # train
    seq_root = "../DETRAC-dataset/DETRAC-train-data/Insight-MVT_Annotation_Train/"  # 图片
    xml_root = "../DETRAC-dataset/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML/"  # 原始xml标注
    label_root = "../DETRAC-dataset-yolo/tmp/train_detrac_txt/"  # 新生成的标签保存目录
    xml2txt(seq_root, xml_root, label_root)

    # # test
    # seq_root = "../DETRAC-dataset/DETRAC-test-data/Insight-MVT_Annotation_Test/"  # 图片
    # xml_root = "../DETRAC-dataset/DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML/"  # 原始xml标注
    # label_root = "../DETRAC-dataset-yolo/tmp/test_detrac_txt/"  # 新生成的标签保存目录
    # xml2txt(seq_root, xml_root, label_root)


if __name__ == "__main__":
    main()
