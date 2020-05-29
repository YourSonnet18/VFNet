import os
import os.path as path
import pickle
import numpy as np
import lmdb
import cv2
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="../datasets/Test2/")
parser.add_argument("--output",type=str, default="../datasets/Test2")
parser.add_argument("--name", type=str, default="Test2")

opt = parser.parse_args()

## create lmdb environment
alist = os.listdir(path.join(opt.dataset,"LR"))
def make_env(R):
    print("making env for",R)
    filenames = []
    for x in alist:
        filenames += [path.join(opt.dataset,R,x,y) for y in os.listdir(path.join(opt.dataset,R,x))]
    data_size_per_img= cv2.imread(filenames[0]).nbytes
    print(data_size_per_img)
    data_size = data_size_per_img*len(filenames)
    env = lmdb.open(path.join(opt.output,R+".lmdb"),map_size=data_size*2)
    txn = env.begin(write=True)
    return txn,env
## create lmdb file
def make_lmdb(folder,R,name_a,txn,keys,resolutions):
    print("making lmdb for "+R,folder)
    filenames = os.listdir(path.join(opt.dataset,R,folder))
    filenames.sort()
    for x in filenames:
        key = "{:03d}_".format(name_a)+path.splitext(x)[0]
        keys.append(key)
        key_byte = key.encode("ascii")
        data = cv2.imread(path.join(opt.dataset,R,folder,x))
        H,W,C = data.shape
        txn.put(key_byte,data, overwrite=True)
        resolutions.append("{:d}_{:d}_{:d}".format(C,H,W))
    return txn,keys,resolutions
txn_lr,env_lr = make_env("LR")
txn_hr,env_hr = make_env("HR")

resolutions_lr = []
resolutions_hr = []
name_a = 0
keys_lr = []
keys_hr = []
for i in alist:
    txn_lr,keys_lr,resolutions_lr = make_lmdb(i,"LR",name_a,txn_lr,keys_lr,resolutions_lr)
    txn_hr,keys_hr,resolutions_hr = make_lmdb(i,"HR",name_a,txn_hr,keys_hr,resolutions_hr)
    name_a += 1
    txn_hr.commit()
    txn_hr = env_hr.begin(write=True)
    txn_lr.commit()
    txn_lr = env_lr.begin(write=True)
txn_lr.commit()
txn_hr.commit()
env_lr.close()
env_hr.close()

def make_meta(R,keys,resolutions):
    print("making meta for "+R)
    meta_info = {"name":opt.name+R}
    if len(set(resolutions))<=1:
        meta_info["resolution"] = [resolutions[0]]
    else:
        meta_info["resolution"] = resolutions
    meta_info["keys"] = keys
    pickle.dump(meta_info,open(path.join(opt.output,R+".lmdb","meta_info.pkl"),"wb"))
make_meta("LR",keys_lr,resolutions_lr)
make_meta("HR",keys_hr,resolutions_hr)