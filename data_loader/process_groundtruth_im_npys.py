import os
import shutil
import sys

import cv2
import numpy as np
from tqdm import tqdm


def create_symbolics():
    npyRoot = "/home/gph/JZSSD/Dataset/ecoco_depthmaps/eval/npy"
    npyIndFile = "eval_data.txt"
    datasetName = None
    with open(os.path.join(npyRoot, npyIndFile), 'r') as fp:
        datas = fp.readlines()
    for data in tqdm(datas):
        if datasetName is not None and not datasetName in data:
            continue
        npyfilesrc = os.path.join(npyRoot, data.strip(), 'images_org.npy')
        npyfiledst = os.path.join(npyRoot, data.strip(), 'images.npy')
        if os.path.exists(npyfiledst):
            os.remove(npyfiledst)
        os.system(f"ln -s {npyfilesrc} {npyfiledst}")


if __name__ == '__main__':
    create_symbolics()