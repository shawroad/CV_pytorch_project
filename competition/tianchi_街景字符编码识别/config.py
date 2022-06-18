"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-18$
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/', type=str, help='数据集路径')
    parser.add_argument('--nums_epoch', default=1, type=int, help='训练轮次')
    parser.add_argument('--batch_size', default=2, type=int, help='批次大小')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='学习率')
    return parser.parse_args()



