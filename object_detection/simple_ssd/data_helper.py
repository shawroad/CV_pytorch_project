"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-18$
"""
"""python
    PascalVOCDataset具体实现过程
"""
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform


class PascalVOCDataset(Dataset):
    def __init__(self, data_folder, split, keep_difficult=False):
        self.split = split.upper()  # 保证输入为纯大写字母，便于匹配{'TRAIN', 'TEST'}
        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)  # n_objects代表当前图片的目标数
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.BoolTensor(objects['difficulties'])  # (n_objects)

        # 如果self.keep_difficult为False,即不保留difficult标志为True的目标
        if not self.keep_difficult:
            boxes = boxes[~difficulties]   # 相当于把不是difficulties标位1 即取到
            labels = labels[~difficulties]
            difficulties = difficulties[~difficulties]

        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)
        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    images, boxes, labels, difficulties = [], [], [], []
    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
        difficulties.append(b[3])
    # (3,224,224) -> (N,3,224,224)
    images = torch.stack(images, dim=0)
    return images, boxes, labels, difficulties  # tensor (N, 3, 224, 224), 3 lists of N tensors each

