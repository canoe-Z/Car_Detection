'''
code by CANOE@2021/06/04
code by zzg@2021/05/10
'''
import os.path as osp
import os
import shutil
from tqdm import tqdm
from common import mkdirs


def rename_image(src_dir, dst_dir):
    mkdirs(dst_dir)
    pwd=os.getcwd()
    abs_src_path = os.path.abspath(src_dir)
    seqs = os.listdir(abs_src_path)

    for seq in tqdm(seqs):
        os.chdir(abs_src_path)
        fileList = os.listdir(seq)

        os.chdir(seq)
        for fileName in fileList:
            image_name = fileName.split(".")[0]
            os.rename(fileName, ('{}_{}.jpg'.format(image_name, str(seq))))

    print("Rename Finished!")

    os.chdir(pwd)
    for seq in tqdm(seqs):
        path = osp.join(abs_src_path, seq)

        for _, _, files in os.walk(path):
            files = sorted(files)
            for i in range(len(files)):
                if i % 10 == 0:
                    if (files[i][-3:] == 'jpg' or files[i][-3:] == 'JPG') or files[i][-4:] == 'jpeg':
                        file_path = path + '/' + files[i]
                        new_file_path = dst_dir + '/' + files[i]
                        shutil.copy(file_path, new_file_path)
    print("Copy Finished!")


def main():
    # train
    src_dir = "../DETRAC-dataset/DETRAC-train-data/Insight-MVT_Annotation_Train/"
    dst_dir = "../DETRAC-dataset-yolo/train"
    rename_image(src_dir, dst_dir)

    # test
    src_dir = "../DETRAC-dataset/DETRAC-test-data/Insight-MVT_Annotation_Test/"
    dst_dir = "../DETRAC-dataset-yolo/test"
    rename_image(src_dir, dst_dir)


if __name__ == "__main__":
    main()
