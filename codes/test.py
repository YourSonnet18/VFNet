import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_lmdb_color import *
import os
import time
import argparse
from utils import *
import logging
import traceback
from logging.handlers import RotatingFileHandler
from pytorch_msssim import ms_ssim

parser = argparse.ArgumentParser()
parser.add_argument('--nFrames', type=int, default=5)
parser.add_argument('--groups', type=int, default=8)
parser.add_argument('--front_RBs',type=int, default=5)
parser.add_argument('--back_RBs',type=int,default=10)
parser.add_argument('--model_folder', default='/share/data/zhaoyanchao/experiment1', help='Location to save checkpoint models')
parser.add_argument('--pretrained_sr', default='epoch_54.pth', help='sr pretrained base model')
parser.add_argument('--result_folder', default='/share/data/zhaoyanchao/results/test1')
parser.add_argument('--test_dir', type=str, default='/dataset/zyc/VF_Validation')
opt = parser.parse_args()
# from pytorch_msssim import ms_ssim
import cv2
def single_forward(model, inp):
  """PyTorch model forward (single test), it is just a simple warpper
  Args:
      model (PyTorch model)
      inp (Tensor): inputs defined by the model

  Returns:
      output (Tensor): outputs of the model. float, in CPU
  """
  with torch.no_grad():
    model_output = model(inp)
    if isinstance(model_output, list) or isinstance(model_output, tuple):
      output = model_output[0]
    else:
      output = model_output
  output = output.data.float().cpu()
  return output


def flipx4_forward(model, inp):
  """Flip testing with X4 self ensemble, i.e., normal, flip H, flip W, flip H and W
  Args:
      model (PyTorch model)
      inp (Tensor): inputs defined by the model

  Returns:
      output (Tensor): outputs of the model. float, in CPU
  """
  # normal
  output_f = single_forward(model, inp)

  # flip W
  output = single_forward(model, torch.flip(inp, (-1, )))
  output_f = output_f + torch.flip(output, (-1, ))
  # flip H
  output = single_forward(model, torch.flip(inp, (-2, )))
  output_f = output_f + torch.flip(output, (-2, ))
  # flip both H and W
  output = single_forward(model, torch.flip(inp, (-2, -1)))
  output_f = output_f + torch.flip(output, (-2, -1))

  return output_f / 4


# parameters
batch_size=2


print("Making model...")
from VFNet import VFNet
model = VFNet(nf=64, nframes=opt.nFrames, groups=opt.groups,front_RBs=opt.front_RBs,back_RBs=opt.back_RBs,upscale_factor=4)
model = model.cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(os.path.join(opt.model_folder,opt.pretrained_sr)))
print("Preparing data...")
dataset = VideoDataset(opt.test_dir,opt.nFrames)
dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False)
print("Begin testing...")
total_frames = 0
total_psnr = 0
# cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(os.path.join(opt.result_folder,"car.mp4"),-1,25,(1280,720))
# total_ms_ssim = 0
with torch.no_grad():
    for i,batch in enumerate(dataloader,1):
        inputs,target = batch[0].float(),batch[1].float()
        inputs = inputs.cuda()

        prediction = flipx4_forward(model,inputs)
        for j in range(batch_size):
            psnr = calculate_psnr(prediction[j],target[j])
            total_psnr += psnr
            total_frames += 1
            img = prediction[j].squeeze().cpu().detach().numpy()
            img = img.transpose((1,2,0)).clip(0,255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #   cv2.imwrite('../results/Car_ensemble/{:06d}.png'.format(total_frames),img)
            video.write(img)
            print("Processing Frame {},PSNR={:.4f}".format(total_frames,psnr))
    print("Processed {} frames, Avg.PSNR={:.4f}".format(total_frames, total_psnr/total_frames))