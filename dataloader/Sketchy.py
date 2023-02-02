import os
import torch
import numpy as np
from torch.utils.data import Dataset

class SketchyDataset(Dataset):
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.data_num= len(os.listdir(self.data_dir))

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        data_path = self.data_dir+f'{idx}.npy'
        video=np.load(data_path,allow_pickle=True)
        video=torch.Tensor(np.moveaxis(video,3,1))
        return video,idx


