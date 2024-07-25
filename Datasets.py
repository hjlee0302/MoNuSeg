import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

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
        png_lbl = Image.open(os.path.join(self.lbl_dir, self.img_list[idx][:-4] + '_label.png'))
        transform = transforms.Pad(padding, fill=0)
        png_lbl = transform(png_lbl)
        np_lbl = np.array(png_lbl)
       # print('img: ', self.img_list[idx])
       # print('lbl: ', self.lbl_list[idx])
       # print('Unique ', np.unique(np_lbl))

 #       cnt = 0
       # print('npshape ', np_lbl.shape)
        for i in range(256):
            for j in range(256):
                if np_lbl[i][j] != 0:
                    np_lbl[i][j] = 1
 #                   cnt = cnt + 1
 #       print(cnt)
        transform = transforms.ToTensor()
        label = transform(np_lbl)
        label = label * 255
        label = label.squeeze(0)
        label = label.long()
#        print(torch.unique(label))
        #print('Label tensor shape:', label.shape)
        #print('Unique values in label tensor after conversion:', torch.unique(label))
        output = {'img': img, 'label': label}

        return output

#data = BratsDataset_seg('/media/NAS/nas_32/hojun/MoNuSeg_data')
#pic = data[0]['label']
#cnt = 0
#for i in range(256):
#    for j in range(256):
#        if pic[i][j] == 1:
#            cnt = cnt + 1
#print('final',cnt)
       # print(f"Pixel value at ({i}, {j}): {pic[i][j]}")
