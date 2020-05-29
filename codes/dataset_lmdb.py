import lmdb
import cv2
import numpy as np
import torch.utils.data as data
import pickle
import os.path as path
import os


class VideoDataset(data.Dataset):
    def __init__(self,image_dir,nFrames):
        super(VideoDataset).__init__()
        self.image_dir = image_dir
        self.nFrames = nFrames
        self.keys_lr, self.resolutions_lr = self.load_meta("LR")
        self.keys_hr, self.resolutions_hr = self.load_meta("HR")
    def load_meta(self,R):
        meta_info = pickle.load(open(path.join(self.image_dir,R+".lmdb","meta_info.pkl"),"rb"))
        keys = meta_info["keys"]
        resolutions = meta_info["resolution"]
        if len(resolutions)==1:
            resolutions = resolutions*len(keys)
        return keys,resolutions
    def load_img(self,env,key,size):
        with env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        img = np.frombuffer(buf,dtype=np.uint8)
        C,H,W = size.split('_')
        img.resize(int(H),int(W),int(C))
        return img
    def __getitem__(self,index):
        env_lr = lmdb.open(path.join(self.image_dir,"LR.lmdb"),readonly=True,
                                     lock=False,readahead=False,meminit=False)
        env_hr = lmdb.open(path.join(self.image_dir,"HR.lmdb"),readonly=True,
                                     lock=False,readahead=False,meminit=False)
        key = self.keys_lr[index]
        target = cv2.cvtColor(self.load_img(env_hr,key,self.resolutions_hr[index]),cv2.COLOR_BGR2RGB).transpose((2,0,1))
        input = cv2.cvtColor(self.load_img(env_lr,key,self.resolutions_lr[index]),cv2.COLOR_BGR2RGB).transpose((2,0,1))
        inputs = []
        name_a,name_b = key.split('_')
        tt = int(self.nFrames/2)
        if self.nFrames&1==0:
            seq = [x for x in range(-tt,tt)]
        else:
            seq = [x for x in range(-tt,tt+1)]
        for i in seq:
            index1 = int(name_b)+i
            key1 = name_a+"_{:06d}".format(index1)
            if key1 in self.keys_lr:
                input1 = self.load_img(env_lr,key1,self.resolutions_lr[index])
                input1 = cv2.cvtColor(input1,cv2.COLOR_BGR2RGB).transpose((2,0,1))
                inputs.append(input1)
            else:
                print("neigbor frame-{} is not exist".format(key1))
                inputs.append(input)
        inputs = np.array(inputs)
        return inputs,target
    def __len__(self):
        return len(self.keys_lr)