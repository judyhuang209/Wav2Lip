from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, traceback
import dlib, json, subprocess
from tqdm import tqdm
from glob import glob
import torch

sys.path.append('../')
import face_detection
from concurrent.futures import ThreadPoolExecutor, as_completed


parser = argparse.ArgumentParser(description='Code to get intersection image files.')

parser.add_argument('--data_root_least', help='Root folder of the detected faces with least amount.', default='frames_only_face_all/', type=str)
parser.add_argument('--data_root_target', help='Target folder of the detected faces to examine.', default='frames_all/', type=str)
parser.add_argument('--num_workers', help='The number of CPUs loading data.', default='8', type=int)

args = parser.parse_args()

def main(args):
    filelist_least = glob(path.join(args.data_root_least, '*.jpg'))
    filelist_target = glob(path.join(args.data_root_target, '*.jpg'))
    filelist_least_name, filelist_target_name = [], []
    for filename in filelist_least:
        filelist_least_name.append(path.basename(filename))
    for filename in filelist_target:
        filelist_target_name.append(path.basename(filename))
    inter = set(filelist_least_name).intersection(set(filelist_target_name))
    print(len(inter))

if __name__ == '__main__':
    main(args)