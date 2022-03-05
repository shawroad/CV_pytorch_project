"""
# -*- coding: utf-8 -*-
# @File    : train.py
# @Time    : 2020/11/26 11:18 上午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import os
import torch
from torchvision import transforms
import numpy as np
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score
from model import Siamese
from mydataset import OmniglotDataSet
from config import set_args
from rlog import rainbow
#
# import torch.nn.functional as F
# F.sigmoid()

def evaluate():
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    sum_step = 0
    loss = 0
    for step, (img1, img2, label) in enumerate(trainLoader):
        sum_step += 1
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        with torch.no_grad():
            output = model(img1, img2)
        loss += loss_fn(output, label)

        zero_m = torch.zeros(label.size())
        one_m = torch.ones(label.size())
        output = torch.where(output > 0.5, one_m, zero_m)
        pred = output.data.cpu().numpy()
        label = label.cpu().numpy()

        labels_all = np.append(labels_all, label)
        predict_all = np.append(predict_all, pred)
    eval_loss = loss / sum_step
    eval_accuracy = accuracy_score(labels_all, predict_all)
    eval_recall = recall_score(labels_all, predict_all)
    eval_precision = precision_score(labels_all, predict_all)
    s = 'epoch:{}, eval_loss: {}, eval_accuracy:{}, eval_precision: {}, eval_recall:{}'.format(
        epoch, eval_loss, eval_accuracy, eval_precision, eval_recall)

    print(s)
    s += '\n'
    with open('result_eval.txt', 'a+') as f:
        f.write(s)
    return eval_loss, eval_accuracy


if __name__ == '__main__':
    args = set_args()
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    # 对数据进行的处理
    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])
    trainSet = OmniglotDataSet(args.train_path, transform=data_transforms, num_sample=1000)
    testSet = OmniglotDataSet(args.test_path, num_sample=100)
    testLoader = DataLoader(testSet, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # 定义损失函数
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model = Siamese()
    model.to(device)
    model.train()

    # 定义优化器
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_acc = 0
    for epoch in range(args.EPOCHS):
        for step, (img1, img2, label) in enumerate(trainLoader):
            # print(img1.size())   # torch.Size([2, 1, 105, 105])
            # print(img2.size())   # torch.Size([2, 1, 105, 105])
            # print(label)  # torch.Size([2, 1])
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output = model(img1, img2)
            loss = loss_fn(output, label)
            zero_m = torch.zeros(label.size())
            one_m = torch.ones(label.size())
            output = torch.where(output > 0.5, one_m, zero_m)
            pred = output.data.cpu().numpy()
            label = label.cpu().numpy()
            acc = accuracy_score(pred, label)
            s = 'Epoch: {}, Step: {}, loss: {:10f}, accuracy:{:10f}'.format(epoch, step, loss, acc)
            rainbow(s)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 验证模型
        eval_loss, eval_accuracy = evaluate()
        if eval_accuracy > best_acc:
            best_acc = eval_accuracy
            # 保存模型
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            os.makedirs(args.save_model, exist_ok=True)
            output_model_file = os.path.join(args.save_model, "best_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

        # 每一轮保存一次模型
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.save_model, "epoch{}_ckpt.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)


