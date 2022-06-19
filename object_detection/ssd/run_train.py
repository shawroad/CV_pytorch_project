"""
@file   : run_train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-18$
"""
import os
import torch
import json
from config import set_args
from model import Model, MultiBoxLoss
from torch.utils.data import DataLoader
from data_helper import PascalVOCDataset, collate_fn
from utils import adjust_learning_rate, save_checkpoint


if __name__ == '__main__':
    args = set_args()

    label_map = json.load(open(os.path.join(args.data_folder, 'label_map.json')))
    model = Model(n_classes=len(label_map))

    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    # Custom dataloaders
    train_dataset = PascalVOCDataset(args.data_folder, split='train', keep_difficult=args.keep_difficult)
    # num_workers=args.workers, pin_memory=True
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # 调整学习率
    decay_lr_at = [150, 190]  # decay learning rate after these many epochs
    for epoch in range(args.nums_epoch):
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, 0.1)
        model.train()  # training mode enables dropout

        for step, batch in enumerate(train_loader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            images, boxes, labels, _ = batch

            predicted_locs, predicted_scores = model(images)  # (N, 441, 4), (N, 441, n_classes)
            # print(predicted_locs.size())    # torch.Size([2, 441, 4])
            # print(predicted_scores.size())   # torch.Size([2, 441, 21])

            loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
            print('epoch:{}, step:{}, loss:{:10f}'.format(epoch, step, loss))
            # Backward prop.
            optimizer.zero_grad()
            loss.backward()

            # Update model
            optimizer.step()

        save_checkpoint(epoch, model, optimizer)



