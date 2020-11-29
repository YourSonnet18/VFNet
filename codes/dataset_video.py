import torch.utils.data as data
import torch
import os
import cv2
import numpy as np
import random
from os import path


class VideoDataset(data.Dataset):
    def __init__(self, file_path, nFrames, GT=False):
        super(VideoDataset, self).__init__()
        cap = cv2.VideoCapture(file_path)
        self.fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames = []
        ret, im = cap.read()
        while ret:
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB).transpose((2,0,1))
            self.frames.append(im)
            ret, im = cap.read()
        self.nFrames = nFrames
      
    def __getitem__(self, index):
        
        input = self.frames[index]
        inputs = []
        
        tt = int(self.nFrames/2)
        if self.nFrames&1==0:
            seq = [x for x in range(-tt,tt)]
        else:
            seq = [x for x in range(-tt,tt+1)]
        for i in seq:
            index1 = index+i
            if index1>=0 and index1<self.fcount:
                input1 = self.frames[index1]
                inputs.append(input1)
            else:
                print("neigbor frame-{} is not exist".format(index1))
                inputs.append(input)
        inputs = np.array(inputs)
        return inputs
    def __len__(self):
        return len(self.frames)

