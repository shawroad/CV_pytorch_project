"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-18$
"""
from torch import nn
from torchvision import models
import os


os.environ['TORCH_HOME'] = './pretrain_model'   # 将预训练模型下载到指定位置


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        model_conv = models.resnet18(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])   # [:-1]是扔掉原始的分类头
        self.cnn = model_conv
        # 加五个分类头  分别预测字符
        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)

    def forward(self, img):
        feat = self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5
