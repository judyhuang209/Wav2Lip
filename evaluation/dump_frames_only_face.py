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
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
# parser.add_argument("--data_root", help="Root folder of the evalutaion videos", required=True)
parser.add_argument("--data_root", help="Root folder of the evalutaion videos", default="/home/judy/Wav2Lip/results/ground_truth/lrs2_gt/")
parser.add_argument("--img_wid", help="Width(X) of cropped faces (X^2)", default=96, type=int)


args = parser.parse_args()

fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
                                   device='cuda:{}'.format(id)) for id in range(args.ngpu)]

face_size = (args.img_wid, args.img_wid)
# template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'


# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

def process_video_file(vfile, args, gpu_id):
    video_stream = cv2.VideoCapture(vfile)

    vidname = os.path.basename(vfile).split('.')[0]
    # fulldir = path.join(args.data_root, "frames_only_face/", vidname)
    fulldir = path.join(args.data_root, "frames_only_face_all/")

    os.makedirs(fulldir, exist_ok=True)
    frames = []
    # count = 0
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)
        # cv2.imwrite(path.join(fulldir, '{}.jpg'.format(count)), frame)
        # count += 1

    batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]

    i = -1
    for fb in batches:
        preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            i += 1
            if f is None:
                continue

            x1, y1, x2, y2 = f
            face_to_save = cv2.resize(fb[j][y1:y2, x1:x2], face_size)
            # cv2.imwrite(path.join(fulldir, '{}_{}.jpg'.format(vidname, i)), fb[j][y1:y2, x1:x2])
            cv2.imwrite(path.join(fulldir, '{}_{}.jpg'.format(vidname, i)), face_to_save)


def mp_handler(job):
    vfile, args, gpu_id = job
    try:
        process_video_file(vfile, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


def main(args):
    print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

    filelist = glob(path.join(args.data_root, '*.mp4'))

    jobs = [(vfile, args, i % args.ngpu) for i, vfile in enumerate(filelist)]
    p = ThreadPoolExecutor(args.ngpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]


if __name__ == '__main__':
    main(args)
