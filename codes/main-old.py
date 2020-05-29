import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import *
import os
import time
import argparse
from utils import *
import logging
from logging.handlers import RotatingFileHandler
#Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--shuffle', action="store_true",default=False)
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

logger =  logging.getLogger("base")
logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(message)s")
handler = RotatingFileHandler(opt.save_folder+"log.txt",maxBytes=10*1024,backupCount=5)
# sHandler = logging.StreamHandler()
logger.addHandler(handler)
# logger.addHandler(sHandler)


logger.info('===> Loading datasets')
train_set = VideoDataset(opt.train_dir,opt.nFrames,opt.file_list, False)
train_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=opt.shuffle)
val_set = VideoDataset(opt.val_dir,opt.nFrames,isVal=True)
val_data_loader = DataLoader(dataset=val_set,batch_size=1,shuffle=True)
val_data_iter = iter(val_data_loader)

logger.info('===> Building model')
if opt.upscale_factor==4:
    from VFNet import VFNet
else:
    from VFNet_x2 import VFNet
model = VFNet(nf=64, nframes=opt.nFrames, groups=opt.groups,front_RBs=opt.front_RBs,back_RBs=opt.back_RBs)
if torch.cuda.is_available() and opt.gpu_mode:
    model = model.cuda()
## utils

loss_function = nn.L1Loss()
optimizer = optim.Adam(model.parameters(),lr=1e-6,betas=(0.9,0.99))
epochs = opt.nEpochs
total_steps = 0


if opt.pretrained:
    start_epoch = int(opt.pretrained_sr[6:-4])
    model_name = os.path.join(opt.save_folder+opt.pretrained_sr)
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc:storage))
        logger.info("Pre_trained model is loaded.")
else:
    start_epoch = 0
def train(epoch):
    epoch_loss = 0
    global total_steps
    model.train()
    for it,batch in enumerate(train_data_loader,1):
        inputs, target = batch[0].type(torch.FloatTensor),batch[1].type(torch.FloatTensor)
        optimizer.zero_grad()
        if torch.cuda.is_available() and opt.gpu_mode:
            inputs = inputs.cuda()
            target = target.cuda()
        t0 = time.perf_counter()
        predicton = model(inputs)
        loss = loss_function(predicton,target)
        loss.backward()
        optimizer.step()
        t1 = time.perf_counter()
        epoch_loss += loss.data
        total_steps +=1
        if total_steps/100%opt.snapshots==0:
            checkpoint(epoch)
            with torch.no_grad():
                validation(epoch,model,total_steps)
        logger.info("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch,it,len(train_data_loader),loss.item(),(t1-t0)))
    temp_psnr = calculate_psnr(predicton[0],target[0])
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.4f} || temp_PSNR: {:.4f}".format(epoch, epoch_loss / len(train_data_loader), temp_psnr))

def validation(epoch,model,total_steps):
    val_inputs, val_target = next(val_data_iter)
    val_inputs = val_inputs.cuda().float()
    val_prediction = model(val_inputs)#B=1,C,H,W
    save_image(val_prediction[0],opt.save_folder+"Epoch{}it{}_SR.png".format(epoch,total_steps))
    save_image(val_target[0],opt.save_folder+"Epoch{}it{}_HR.png".format(epoch,total_steps))
    save_image(val_inputs[0,opt.nFrames//2,:,:,:],opt.save_folder+"Epoch{}it{}_LR.png".format(epoch,total_steps))
    psnr = calculate_psnr(val_prediction[0],val_target[0])
    logger.info("Validation PSNR:{:.4f}".format(psnr))
def checkpoint(epoch):
    model_out_path = opt.save_folder+"epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(),model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

logger.info('===> Sart Training')
validation(0,model,0)
for epoch in range(start_epoch,epochs):
    try :
        train(epoch)
    except Exception as e:
        logger.error(e)
        checkpoint(epoch)
        break
    checkpoint(epoch)
    if (epoch+1)%(opt.snapshots)==0:
        for param_group in optimizer.param_groups:
            param_group["lr"] /= 2
        logger.info("Learning rate decay: lr={}".format(optimizer.param_groups[0]["lr"]))

    