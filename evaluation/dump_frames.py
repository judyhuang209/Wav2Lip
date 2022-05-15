import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('../face_detection/detection/sfd/s3fd.pth'):
    raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
							before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob

sys.path.insert(0, '..')
import face_detection

parser = argparse.ArgumentParser()
parser.add_argument('--ncpu', help='Number of CPUs across which to run in parallel', default=8, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument("--data_root", help="Root folder of the evalutaion videos", required=True)

args = parser.parse_args()

# template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'


# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

def process_video_file(vfile, args, gpu_id):
    video_stream = cv2.VideoCapture(vfile)

    vidname = os.path.basename(vfile).split('.')[0]
    fulldir = path.join(args.data_root, "frames/", vidname)
    os.makedirs(fulldir, exist_ok=True)
    frames = []
    count = 0
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)
        cv2.imwrite(path.join(fulldir, '{}.jpg'.format(count)), frame)
        count += 1


def mp_handler(job):
    vfile, args, gpu_id = job
    try:
        process_video_file(vfile, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


def main(args):
    print('Started processing for {} with {} CPUs'.format(args.data_root, args.ncpu))

    filelist = glob(path.join(args.data_root, '*.mp4'))

    jobs = [(vfile, args, i % args.ngpu) for i, vfile in enumerate(filelist)]
    p = ThreadPoolExecutor(args.ncpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]


if __name__ == '__main__':
    main(args)
