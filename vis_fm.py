from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# from torchvision.models import resnet50
from torch.utils import data as data_utils
from models.wav2lip_coordatt_vis import Wav2Lip as Wav2Lip
import argparse, random, os, cv2
from os.path import dirname, join, basename, isfile
import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from glob import glob
from hparams import hparams, get_image_list
import audio
import numpy as np

'''
DataLoader = [x, indiv_mels, mel, gt]
'''
parser = argparse.ArgumentParser(description='Code to visualize the feature maps of Wav2Lip model')
parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default='lrs2_preprocessed/', type=str)
parser.add_argument('--vis_dir', help='Save visualizations to this directory', default='results/vis/', type=str)
parser.add_argument('--batch_size', help='batch_size', default=1, type=int)
parser.add_argument('--checkpoint_path', help='Use this checkpoint', default='my_checkpoints/eval/new_coordattn/eval_checkpoint_step000109500.pth', type=str)

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('use_cuda: {}'.format(use_cuda))

global_step = 0
global_epoch = 0
syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx: end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1  # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue

            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2] // 2:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return x, indiv_mels, mel, y

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

def save_sample_images(x, gt, visualization, global_step, vis_dir, i, f_i):
    print(x.shape)
    print(type(x))
    # print(type(g))
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    # g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    visualization = (visualization * 255.).astype(np.uint8)
    refs, inps = x[..., 3:], x[..., :3]

    print('refs', refs.shape)
    print('vis', visualization.shape)
    folder = join(vis_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, gt), axis=-2)
    # print(collage.shape)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])
        # cv2.imwrite('{}/{}_vis_{}_{}.jpg'.format(folder, batch_idx, i, f_i), visualization)

def save_feature_maps(img, g, global_step, vis_dir):
    g = (g.detach().cpu().numpy() * 255.).astype(np.uint8)
    folder = join(vis_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    for t in range(len(g)):
        for m in range(len(g[t])):
            print(t, m, g[t][m][0][0])
            heatmap = cv2.applyColorMap(g[t][m], cv2.COLORMAP_JET)
            # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = np.float32(heatmap) / 255
            # heatmap = heatmap - np.mean(heatmap)
            # heatmap = heatmap / np.std(heatmap)
            cam = heatmap + img
            cam = cam / np.max(cam)
            cam = np.uint8(255. * cam)

            cv2.imwrite('{}/t{}_map_m{}.jpg'.format(folder, t, m), cam)
            cv2.imwrite('{}/t{}_map_m{}_00.jpg'.format(folder, t, m), g[t][m])
            img1 = cv2.imread('{}/t{}_map_m{}.jpg'.format(folder, t, m))
            img2 = cv2.imread('{}/t{}_map_m{}_00.jpg'.format(folder, t, m))
            img3 = np.uint8(255. * img)
            con = np.concatenate((img2, img1, img3), axis=1)
            cv2.imwrite('{}/t{}_map_m{}_con.jpg'.format(folder, t, m), con)

            


if __name__ == "__main__":
    vis_dir = args.vis_dir
    # model = resnet50(pretrained=True)
    model = Wav2Lip().to(device)

    # print(model)
    # target_layers = [model.layer4[-1]]

    # Dataset and Dataloader setup
    test_dataset = Dataset('val')

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=4)

    for batch in test_data_loader:
        #print(batch)
        x, indiv_mels, mel, gt = batch
        # Create an input tensor image for your model..
        break
    del test_data_loader
    # Move data to CUDA device
    x = x.to(device)
    # mel = mel.to(device)
    indiv_mels = indiv_mels.to(device)
    gt = gt.to(device)
    print('size check')
    # print('mel', indiv_mels.size())
    # print('x', x.size())
    # print(gt.size())

    a_t = indiv_mels.view(indiv_mels.size(0), -1)
    x_t = x.view(x.size(0), -1)
    print('x_t', x_t.size())
    # x_t = x_t.view(x.size(0), 6, syncnet_T, hparams.img_size, hparams.img_size)
    # a_t = a_t.view(indiv_mel.size(0), syncnet_T, 1, 80, 16)
    print('a_t', a_t.size())

    # print(torch.equal(x, x_t))
    input_tensor = torch.cat((a_t, x_t), dim=-1)
    # aa, xx = torch.split(input_tensor, [syncnet_T*80*16, 6*syncnet_T*hparams.img_size*hparams.img_size], dim=-1)
    # print('CHECKING...')
    # print(torch.equal(aa, a_t))
    # print(torch.equal(xx, x_t))
    # print('aa', aa.size())

    print('input', input_tensor.shape)
    # print(mel.size(0)) # args.batch_size
    # print(gt.size(0))

    # Note: input_tensor can be a batch tensor with several images!
    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=hparams.initial_learning_rate)
    optimizer.zero_grad()
    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)
    target_layers = []
    rgb_img = x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1)
    rgb_img = rgb_img[0, 0, :, :, 3:6]

    # for i, target_layer in enumerate(model.face_decoder_blocks):
    #     # Construct the CAM object once, and then re-use it on many images:
    #     # target_layers.append(target_layer)
    #     # print('layer ', i, target_layers)
    #     cam = EigenCAM(model=model, target_layers=target_layer, use_cuda=use_cuda)

    #     # You can also use it within a with statement, to make sure it is freed,
    #     # In case you need to re-create it inside an outer loop:
    #     # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #     #   ...

    #     # We have to specify the target we want to generate
    #     # the Class Activation Maps for.
    #     # If targets is None, the highest scoring category
    #     # will be used for every image in the batch.
    #     # Here we use ClassifierOutputTarget, but you can define your own custom targets
    #     # That are, for example, combinations of categories, or specific outputs in a non standard model.
    #     # targets = [e.g ClassifierOutputTarget(281)]
    #     targets = None
    #     # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.

    #     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    #     grayscale_cam = torch.from_numpy(grayscale_cam)
    #     print('cam type', type(grayscale_cam))
    #     print('cam shape', grayscale_cam.shape)
    #     audio_cam, face_cam = torch.split(grayscale_cam, [syncnet_T*80*16, 6*syncnet_T*hparams.img_size*hparams.img_size], dim=-1)
    #     print('face_cam shape', face_cam.shape)

        # In this example grayscale_cam has only one image in the batch:

        # face_cam =  face_cam.view(-1, 6, syncnet_T, hparams.img_size, hparams.img_size)
        # for f_i in range(face_cam.size(0)):
        #     f_cam = face_cam[f_i:f_i+1, :3, ...]

        #     f_cam = f_cam.detach().cpu().numpy().transpose(0, 2, 3, 4, 1)
        #     print('face_cam2.5 shape', f_cam.shape)
        #     print('rgb_img shape', rgb_img.shape)

        #     f_cam = f_cam[0, 0, :]
        #     print('face_cam3 shape', f_cam.shape)
        #     # print('face_cam', f_cam)
        #     # print('rgb_img', rgb_img)
        #     visualization = show_cam_on_image(rgb_img, f_cam, use_rgb=True)
        #     # print(rgb_img.dtype) # float32
        #     # print(face_cam.dtype) # float32
        #     # print(visualization.dtype) # unit8

        #     save_sample_images(x, gt, visualization, global_step, vis_dir, i, f_i)
    g = model(input_tensor)
    save_feature_maps(rgb_img, g, global_step, vis_dir)