"""
# -*- coding: utf-8 -*-
# @File    : config.py
# @Time    : 2020/11/26 11:18 上午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='./data/images_background', type=str, help="training folder")
    parser.add_argument('--test_path', default='./data/images_evaluation', type=str, help='path of testing folder')
    parser.add_argument('--batch_size', default=16, type=int, help="how much way one-shot learning")
    parser.add_argument('--times', default=400, type=int, help='number of samples to test accuracy')
    parser.add_argument('--workers', default=2, type=int, help='number of dataLoader workers')
    parser.add_argument('--learning_rate', default=0.0003, type=float, help='number of dataLoader workers')
    parser.add_argument('--EPOCHS', default=10, type=int, help='number of dataLoader workers')
    parser.add_argument('--save_model', default='./save_model', type=str, help='path of testing folder')









    # num_train_epochs
    parser.add_argument('--num_train_epochs', default=20, type=str, help='code will operate in this gpu')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")


    args = parser.parse_args()
    return args


