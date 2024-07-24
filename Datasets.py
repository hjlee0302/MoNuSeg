import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class BratsDataset_seg(torch.utils.data.Dataset):
    def __init__(self, root, mode='train'):
        self.root = root
        self.mode = mode
        self.img_dir = os.path.join(self.root, 'images', self.mode)
        self.lbl_dir = os.path.join(self.root, 'labels_instance', self.mode)
        self.img_list = os.listdir(self.img_dir)
        self.lbl_list = os.listdir(self.lbl_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        padding = (3, 3 ,3 ,3)
        png_img = Image.open(os.path.join(self.img_dir, self.img_list[idx]))
        transform = transforms.Pad(padding, fill=0)
        png_img = transform(png_img)
        np_img = np.array(png_img)
        transform = transforms.ToTensor()
        img = transform(np_img)
        png_lbl = Image.open(os.path.join(self.lbl_dir, self.lbl_list[idx]))
        transform = transforms.Pad(padding, fill=0)
        png_lbl = transform(png_lbl)
        np_lbl = np.array(png_lbl)
        transform = transforms.ToTensor()
        label = transform(np_lbl)
        output = {'img': img, 'label': label}

        return output

data = BratsDataset_seg('/home/hojunlee/Desktop/MoNuSeg_oridata')
print(data[0]['label'])
