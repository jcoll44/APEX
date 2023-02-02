import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import time
import wandb
from PIL import Image
import h5py



class Box2D(Dataset):
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.data_num= 2000
        # self.length = length

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        # data_path = self.data_dir+f'{idx}'+"/"
        # video = torch.zeros(self.length, 3, 128, 128)
        # counter =0
        # for i in range(0,20):
            # img = Image.open(data_path+str(i)+".png").convert("RGB")
            # img = T.ToTensor()(img)
            # img = T.Resize(size=[128])(img)
            # # wandb.log({"Image": wandb.Image(data_path+str(i)+".png")})
            # video[counter,:,:,:] = img
            # counter = counter+1
        hf = h5py.File(self.data_dir+str(idx)+".h5", 'r')
        # print(hf.keys())
        video = np.array(hf.get('video'))
        # print(video.shape)
        video_torch = torch.from_numpy(video)
        video_torch = video_torch / 255
        hf.close()

        return video_torch.float() 


