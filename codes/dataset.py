import torch.utils.data as data
import torch
import os
import cv2
import numpy as np
import random
from os import path


class VideoDataset(data.Dataset):
    def __init__(self, image_dir, nFrames, isVal=False):
        super(VideoDataset, self).__init__()
        
        alist = os.listdir(path.join(image_dir,"LR"))
        self.LR_filenames = []
        self.HR_filenames = []
        for x in alist:
            self.LR_filenames += [path.join(image_dir,"LR",x,y) for y in os.listdir(path.join(image_dir,"LR",x))]
            self.HR_filenames += [path.join(image_dir,"HR",x,y) for y in os.listdir(path.join(image_dir,"HR",x))]
        self.nFrames = nFrames
        self.image_dir = image_dir
        self.isVal = isVal
    def __getitem__(self, index):
        filepath = self.LR_filenames[index]
        input = cv2.cvtColor(cv2.imread(filepath),cv2.COLOR_BGR2RGB)
        input = np.transpose(input,(2,0,1))
        inputs = []
        target = cv2.cvtColor(cv2.imread(self.HR_filenames[index]),cv2.COLOR_BGR2RGB)
        target = np.transpose(target,(2,0,1))
        tt = int(self.nFrames/2)
        if self.nFrames&1==0:
            seq = [x for x in range(-tt,tt)]
        else:
            seq = [x for x in range(-tt,tt+1)]
        for i in seq:
            index1 = int(filepath[-10:-4])+i
            file_name1 = filepath[:-10]+"{:06d}".format(index1)+".png"
            if os.path.exists(file_name1):
                input1 = cv2.cvtColor(cv2.imread(file_name1),cv2.COLOR_BGR2RGB).transpose((2,0,1))
                inputs.append(input1)
            else:
                print("neigbor frame-{} is not exist".format(file_name1))
                inputs.append(input)
        inputs = np.array(inputs)
        return inputs, target
    def __len__(self):
        return len(self.LR_filenames)

