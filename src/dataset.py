import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from src.cfg import config



def create_dataframe(image_dir, mask_dir):
    dataframe = []
    img_list = os.listdir(image_dir)
    for i in img_list:
        img_name = i.split('.')[0]
        img_path = image_dir + f'{img_name}.jpg'
        mask_path = mask_dir + f'{img_name}.png'
        image = os.path.join(image_dir, i)
        img = cv2.imread(image)
        height, width = img.shape[0], img.shape[1]
        dataframe.append((img_name, img_path, mask_path, height, width))

    df = pd.DataFrame(dataframe, columns=['img_id', 'img_path', 'mask_path', 'height', 'width'])
    return df


def img2tensor(img, dtype:np.dtype=np.float32):
        #img = np.array(img) / 255
        if img.ndim == 2:
            img = np.expand_dims(img, 2)
        img = np.transpose(img, (2, 0, 1)) 
        return torch.from_numpy(img.astype(dtype, copy=False))


class Mydataset(Dataset):

    def __init__(self, df, transforms=None):

        self.df = df
        self.image_path = df['img_path']
        self.mask_path = df['mask_path']
        self.transforms = transforms
    
    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = cv2.resize(img, (config.img_sz)) 
        img = img.astype(np.float32) / 255.0  
        
        mask_path = self.mask_path[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  
        mask = cv2.resize(mask, (config.img_sz))
        mask = mask.astype(np.float32) / 255.0  
        
        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        img = img2tensor(img)
        mask = img2tensor(mask)

        return img, mask

    def __len__(self):
        return len(self.df)
    