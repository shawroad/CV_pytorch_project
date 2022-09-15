"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-09-14
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser('--densenet进行图片分类')

    parser.add_argument('--data_path', default='/Users/xiaolu10/Desktop/data/flower_data/flower_photos', type=str,
                        help='训练数据集 一类图片放入一个文件夹')

    # 数据集下载地址: https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--num_classes', type=int, default=5, help='类别数')
    parser.add_argument('--epochs', type=int, default=30, help='训练多少轮')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--lrf', type=float, default=0.1)

    # https://download.pytorch.org/models/densenet121-a639ec97.pth
    parser.add_argument('--pretrain_weights', type=str, default='./pretrain_weights/densenet121-a639ec97.pth',
                        help='initial weights path')
    parser.add_argument('--freeze_layers', type=bool, default=False, help='是否值微调最后的全连接')
    parser.add_argument('--output_dir', default='./output', type=str, help='模型输出目录')
    return parser.parse_args()