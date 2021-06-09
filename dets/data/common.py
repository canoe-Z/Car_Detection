import os.path as osp
import os


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)
