"""
@file   : run_train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-18$
"""
import glob, json
import numpy as np
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import torchvision.transforms as transforms
import torch.nn as nn
from model import Model
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_helper import SVHNDataset
from config import set_args


def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for batch in enumerate(test_loader):
                if torch.cuda.is_available():
                    batch = (t.cuda() for t in batch)
                input_x, target = batch
                c0, c1, c2, c3, c4 = model(input_x)
                if torch.cuda.is_available():
                    output = np.concatenate([
                        c0.data.cpu().numpy(),
                        c1.data.cpu().numpy(),
                        c2.data.cpu().numpy(),
                        c3.data.cpu().numpy(),
                        c4.data.cpu().numpy()], axis=1)
                else:
                    output = np.concatenate([
                        c0.data.numpy(),
                        c1.data.numpy(),
                        c2.data.numpy(),
                        c3.data.numpy(),
                        c4.data.numpy()], axis=1)
                test_pred.append(output)
        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


def validate(val_loader, model):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            input_x, target = batch
            c0, c1, c2, c3, c4 = model(input_x)
            loss = loss_func(c0, target[:, 0]) + \
                   loss_func(c1, target[:, 1]) + \
                   loss_func(c2, target[:, 2]) + \
                   loss_func(c3, target[:, 3]) + \
                   loss_func(c4, target[:, 4])
            val_loss.append(loss.item())
    return np.mean(val_loss)


if __name__ == '__main__':
    args = set_args()
    # 加载数据
    # train data
    train_path = glob.glob(args.data_path + 'mchar_train/*.png')
    train_path.sort()
    train_json = json.load(open(args.data_path + 'mchar_train.json'))
    train_label = [train_json[x]['label'] for x in train_json]
    print('训练集:', len(train_path), len(train_label))
    train_dataset = SVHNDataset(train_path, train_label, transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.RandomCrop((60, 120)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # 可以指定num_workes加快数据读取

    # val data
    val_path = glob.glob(args.data_path + 'mchar_val/*.png')
    val_path.sort()
    val_json = json.load(open(args.data_path + 'mchar_val.json'))
    val_label = [val_json[x]['label'] for x in val_json]
    print('验证集:', len(val_path), len(val_label))
    val_dataset = SVHNDataset(val_path, val_label, transforms.Compose([
        transforms.Resize((60, 120)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = Model()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if torch.cuda.is_available():
        model.cuda()

    best_loss = 1000.0
    for epoch in range(args.nums_epoch):
        model.train()
        train_loss = []
        for step, batch in enumerate(train_dataloader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            input_x, target = batch
            c0, c1, c2, c3, c4 = model(input_x)

            loss = loss_func(c0, target[:, 0]) + \
                   loss_func(c1, target[:, 1]) + \
                   loss_func(c2, target[:, 2]) + \
                   loss_func(c3, target[:, 3]) + \
                   loss_func(c4, target[:, 4])

            optimizer.zero_grad()
            loss.backword()
            optimizer.step()
            train_loss.append(loss.item())
        average_train_loss = np.mean(train_loss)

        # 验证
        average_val_loss = validate(val_dataloader, model)

        # 预测
        val_predict_label = predict(val_dataloader, model, 1)
        val_predict_label = np.vstack([
            val_predict_label[:, :11].argmax(1),
            val_predict_label[:, 11:22].argmax(1),
            val_predict_label[:, 22:33].argmax(1),
            val_predict_label[:, 33:44].argmax(1),
            val_predict_label[:, 44:55].argmax(1),
        ]).T

        val_label_pred = []
        for x in val_predict_label:
            val_label_pred.append(''.join(map(str, x[x != 10])))

        val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))
        print('Epoch: {0}, Train loss: {1} \t Val loss: {2} \t Val Acc: {3}'.format(
            epoch, average_train_loss, average_val_loss, val_char_acc))

        # 记录下验证集精度
        if average_val_loss < best_loss:
            best_loss = average_val_loss
            torch.save(model.state_dict(), './model.pt')
