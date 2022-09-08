"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-18$
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='../../input_data/VOCdevkit', type=str, help='数据集路径')
    parser.add_argument('--nums_epoch', default=1, type=int, help='训练轮次')
    parser.add_argument('--batch_size', default=2, type=int, help='批次大小')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='学习率')
    parser.add_argument('--workers', default=1, type=int, help='指定用几个进程加载数据')
    parser.add_argument('--keep_difficult', default=True, type=bool, help='是否标记处难检测的目标')

    parser.add_argument('--momentum', default=0.9, type=float, help='')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='权重衰减')
    return parser.parse_args()
