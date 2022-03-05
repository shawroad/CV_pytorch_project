"""
# -*- coding: utf-8 -*-
# @File    : mydataset.py
# @Time    : 2020/11/26 11:26 上午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from numpy.random import choice as npc


class OmniglotDataSet(Dataset):
    # 加载训练集
    def __init__(self, data_path, transform=None, num_sample=None):
        super(OmniglotDataSet, self).__init__()
        np.random.seed(0)
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(data_path)
        self.num_sample = num_sample

    def loadToMem(self, data_path):
        print('开始加载数据...')
        datas = dict()
        agrees = [0, 90, 180, 270]
        idx = 0
        for agree in agrees:
            for alphaPath in os.listdir(data_path):
                for charPath in os.listdir(os.path.join(data_path, alphaPath)):
                    # 此时charPath是每个类别数据的路径
                    datas[idx] = []   # 一个类别 对应一批数据
                    for samplePath in os.listdir(os.path.join(data_path, alphaPath, charPath)):
                        # samplePath是每张图片的路径
                        filePath = os.path.join(data_path, alphaPath, charPath, samplePath)
                        # 加载图片 并进行旋转和切换
                        datas[idx].append(Image.open(filePath).rotate(agree).convert('L'))
                    idx += 1
        print('train datasets load finished!')
        return datas, idx

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            # 相当于直接从一个label下找到两张相同的照片  正样本
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            # 确保idx1和idx2不能相等  也就是两张图片不能相似
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        # 是否需要对数据增广
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))



if __name__ == '__main__':
    # 测试一下
    omniglotTrain = OmniglotDataSet('./data/images_background')
    # print(omniglotTrain)

