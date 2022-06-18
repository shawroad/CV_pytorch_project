"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-18$
"""
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        lbl = np.array(self.img_label[index], dtype=np.int)    # 取出类别  [1 2 8]

        # 最大设置每个上面有5个字符  10算是填充
        lbl = list(lbl) + (5 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:5]))   # 之所以这里还切片 是防止有超过5个字符的

    def __len__(self):
        return len(self.img_path)
