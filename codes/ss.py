import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from VFNet import VFNet
from dataset_lmdb import VideoDataset
from utils import *
import argparse
import time

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', default=True, action="store_false")
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--train_dir', type=str, default='../datasets/REDS')
parser.add_argument('--val_dir', type=str, default='../datasets/dance')
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt')
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--groups', type=int, default=8)
parser.add_argument('--front_RBs',type=int, default=5)
parser.add_argument('--back_RBs',type=int,default=10)
# parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
# parser.add_argument('--data_augmentation', type=bool, default=True)
# parser.add_argument('--model_type', type=str, default='RBPN')
# parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--pretrained_sr', default='3x_dl10VDBPNF7_epoch_84.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='../experiments/', help='Location to save checkpoint models')
# parser.add_argument('--prefix', default='F7', help='Location to save checkpoint models')
opt = parser.parse_args()

def validation(epoch,total_steps):
    with torch.no_grad():
        val_inputs, val_target = next(val_data_iter)
        val_inputs = val_inputs.cuda().float()
        val_prediction = model(val_inputs)#B=1,C,H,W
        save_image(val_prediction[0],opt.save_folder+"Epoch{}it{}_SR.png".format(epoch,total_steps))
        print("image1 saved")
        save_image(val_target[0],opt.save_folder+"Epoch{}it{}_HR.png".format(epoch,total_steps))
        print("image2 saved")
        save_image(val_inputs[0,opt.nFrames//2,:,:,:],opt.save_folder+"Epoch{}it{}_LR.png".format(epoch,total_steps))
        print("image3 saved")
        psnr = calculate_psnr(val_prediction[0],val_target[0])
        print("Validation PSNR:{:.4f}".format(psnr))


print('===> Loading datasets')
# val_set = VideoDataset(opt.val_dir,opt.nFrames,isVal=True)
# val_data_loader = DataLoader(dataset=val_set,batch_size=1,shuffle=True)
# val_data_iter = iter(val_data_loader)
train_set = VideoDataset(opt.train_dir,opt.nFrames)
train_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
# print('===> Building model')
# model = VFNet(nf=64, nframes=5, groups=8,front_RBs=5,back_RBs=10).cuda()

loss_function = nn.L1Loss()
# optimizer = optim.Adam(model.parameters(),lr=1e-6,betas=(0.9,0.99))
# validation(0,10)
for inputs,target in train_data_loader:
    save_image(target[0],opt.save_folder+"Epoch{}it{}_HR.png".format(0,10))
    save_image(inputs[0,opt.nFrames//2,:,:,:],opt.save_folder+"Epoch{}it{}_LR.png".format(0,10))
    save_image(inputs[0,opt.nFrames//2-1,:,:,:],opt.save_folder+"Epoch{}it{}_LR-1.png".format(0,10))
    save_image(inputs[0,opt.nFrames//2+1,:,:,:],opt.save_folder+"Epoch{}it{}_LR+1.png".format(0,10))
    break

# pth = torch.load("epoch_6.pth", map_location=torch.device("cpu"))
# for i in pth.keys():
#     print(i+"  "+str(list(pth[i].size())))

    