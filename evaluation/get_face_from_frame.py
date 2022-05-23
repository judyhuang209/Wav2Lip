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


parser = argparse.ArgumentParser(description='Code to detect face in frame-dumps of generated videos.')

parser.add_argument('--data_root_face', help='Root folder to save the detected faces.', default='frames_only_face_all/', type=str)
parser.add_argument('--data_root_frame', help='Root folder of the frame-dumps of generated videos.', default='frames_all/', type=str)
parser.add_argument('--data_root_gt', help='Root folder of the ground truth video to record coordinates to crop if still cannot detect face.', default='frames_all/', type=str)
parser.add_argument('--num_workers', help='The number of CPUs loading data.', default='8', type=int)
parser.add_argument('--ngpu', help='The number of GPUs.', default='1', type=int)
parser.add_argument('--img_wid', help='Width(X) of cropped faces (X^2)', default=96, type=int)

args = parser.parse_args()

fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
                                   device='cuda:{}'.format(id)) for id in range(args.ngpu)]

face_size = (args.img_wid, args.img_wid)

def process_video_file(d_file, args, gpu_id):
    dump_frame_path = path.join(args.data_root_frame, d_file)
    dump_frame_img = cv2.imread(dump_frame_path)

    frame = []
    frame.append(dump_frame_img)
    frame = [frame]
    i = -1

    for fb in frame:
        preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            i += 1
            if f is None:
                print('Still can\'t detect face in {}'.format(d_file))
                print('Start retrieving face coordinates from ground truth video...')
                gt_path = path.join(args.data_root_gt, d_file)
                gt_img = cv2.imread(gt_path)
                gt_frame = [[gt_img]]

                f = fa[gpu_id].get_detections_for_batch(np.asarray(gt_frame[0]))[0]
                print('Get face from ground truth video at {}.'.format(f))

            x1, y1, x2, y2 = f
            face_to_save = cv2.resize(fb[j][y1:y2, x1:x2], face_size)
            # cv2.imwrite(path.join(fulldir, '{}_{}.jpg'.format(vidname, i)), fb[j][y1:y2, x1:x2])
            cv2.imwrite(path.join(args.data_root_face, d_file), face_to_save)
            print('Saved {}!'.format(path.join(args.data_root_face, d_file)))
            break


def mp_handler(job):
    d_file, args, gpu_id = job
    try:
        process_video_file(d_file, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


def main(args):
    print('Started processing for {} with {} GPUs'.format(args.data_root_face, args.ngpu))

    filelist_face = glob(path.join(args.data_root_face, '*.jpg'))
    filelist_frame = glob(path.join(args.data_root_frame, '*.jpg'))
    filelist_face_name, filelist_frame_name = [], []
    for filename in filelist_face:
        filelist_face_name.append(path.basename(filename))
    for filename in filelist_frame:
        filelist_frame_name.append(path.basename(filename))
    set_differ = set(filelist_frame_name) - set(filelist_face_name)
    list_differ = list(set_differ)
    print(list_differ) # ['12994_1.jpg']

    if list_differ is None:
        print('No missing face frames! Skipping...')
        sys.exit()

    jobs = [(d_file, args, i % args.ngpu) for i, d_file in enumerate(list_differ)]
    p = ThreadPoolExecutor(args.ngpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]


if __name__ == '__main__':
    main(args)
