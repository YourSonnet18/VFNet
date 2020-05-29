import numpy as np
import torch
import math
import cv2

def save_image(tensor,img_path):
    img = tensor.squeeze().cpu().detach().numpy()
    img = img.transpose((1,2,0)).clip(0,255)
    img = cv2.cvtColor(img, cv2.RGB2GRAY)
    cv2.imwrite(img_path,img)

def calculate_psnr(img1,img2):
    with torch.no_grad():
        img1 = img1.squeeze().cpu().detach().float()
        img2 = img2.squeeze().cpu().detach().float()
        mse = torch.mean((img1-img2)**2)
        if mse==0:
            return float("inf")
        return 20*math.log10(255/math.sqrt(mse))