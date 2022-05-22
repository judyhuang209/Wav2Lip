import time, pdb, argparse, subprocess
import glob
import os
from tqdm import tqdm
import torch
import torchvision.transforms as TF
from PIL import Image
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

parser = argparse.ArgumentParser(description='Code to evaluate MS-SSIM and SSIM.')

parser.add_argument("--data_root_gt", help="Root folder of the ground truth video frames.", required=True, type=str)
parser.add_argument("--data_root_gen", help="Root folder of the generated video frames.", required=True, type=str)
parser.add_argument("--batch_size", help="Batch size of calculating MS-SSIM and SSIM", default="50", type=int)
parser.add_argument("--num_workers", help="The number of CPUs loading data.", default="8", type=int)

args = parser.parse_args()

# print(len(all_gt_videos))
# print(len(all_gen_videos))

# for i in range (100):
#     print('gt vid name {}: {}'.format(i, all_gt_videos[i]))
#     print('gen vid name {}: {}'.format(i, all_gen_videos[i]))

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        width, height = img.size
        # resize image to calculate ssim
        if width <= 160:
            img = img.resize((200, 200))
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def main():
    gt_path = os.path.join(args.data_root_gt, "*.jpg")
    gen_path = os.path.join(args.data_root_gen, "*.jpg")

    all_gt_videos_list = sorted(glob.glob(gt_path))
    all_gen_videos_list = sorted(glob.glob(gen_path))

    if args.batch_size > len(all_gt_videos_list):
        print(('Warning: batch size is bigger than the data size. '
                'Setting batch size to data size'))
        batch_size = len(all_gt_videos_list)
    else:
        batch_size = args.batch_size

    assert len(all_gt_videos_list) == len(all_gen_videos_list), 'Datasets are not the same size.'

    gt_dataset = ImagePathDataset(all_gt_videos_list, transforms=TF.ToTensor())
    gen_dataset = ImagePathDataset(all_gen_videos_list, transforms=TF.ToTensor())

    # load datasets in batches
    gt_dataloader = torch.utils.data.DataLoader(gt_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=args.num_workers)
    gen_dataloader = torch.utils.data.DataLoader(gen_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=args.num_workers)

    avg_ssim_val = 0.
    avg_ms_ssim_val = 0.
    
    prog_bar = tqdm(zip(gt_dataloader, gen_dataloader), total=len(gt_dataloader))

    count = 0
    for gt_batch, gen_batch in prog_bar:
        # for videofile_idx in prog_bar:
        # 	videofile = all_videos[videofile_idx]
        # 	offset, confidence, min_distance = s.evaluate(opt, videofile=videofile)
        ssim_val = ssim( gen_batch, gt_batch, data_range=1, size_average=False ) # return (N,)
        ms_ssim_val = ms_ssim( gen_batch, gt_batch, data_range=1, size_average=False ) # (N,)

        avg_ssim_val += torch.sum(ssim_val)
        avg_ms_ssim_val += torch.sum(ms_ssim_val)
        count += len(ssim_val)
        
        prog_bar.set_description('Avg SSIM: {}, Avg MS-SSIM: {}, Count: {}'.format(avg_ssim_val / count, avg_ms_ssim_val / count, count) )
        prog_bar.refresh()
        
        # print ('Average Confidence: {}'.format(avg_confidence/len(all_videos)))
        # print ('Average Minimum Distance: {}'.format(avg_min_distance/len(all_videos)))

        # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
        # Y: (N,3,H,W)
    
    print ('Average SSIM: {}'.format(avg_ssim_val/len(all_gt_videos_list)))
    print ('Average MS SSIM: {}'.format(avg_ms_ssim_val/len(all_gt_videos_list)))

if __name__ == "__main__":
    main()