"""
@file   : run_train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-09-07
"""
import os
import math
import torch
from tqdm import tqdm
from torch import nn
import torch.optim as optim
from sklearn import metrics
from config import set_args
from utils import AverageMeter   # 累计平均
from torchvision import transforms
from model import AlexNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_helper import CLSDataSet, read_split_data
import torch.optim.lr_scheduler as lr_scheduler


@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    eval_predict, eval_targets = [], []
    for batch in tqdm(data_loader):
        if torch.cuda.is_available():
            batch = (t.cuda() for t in batch)
        images, labels = batch
        logits = model(images)
        eval_targets.extend(labels.cpu().detach().numpy().tolist())
        eval_predict.extend(torch.max(logits, dim=1)[1].cpu().detach().numpy().tolist())
    val_accuracy = metrics.accuracy_score(eval_targets, eval_predict)
    return val_accuracy


if __name__ == '__main__':
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path, args.output_dir)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 1])  # 使用多少worker
    print('Using {} workers'.format(nw))
    train_dataset = CLSDataSet(images_path=train_images_path, images_class=train_images_label, transform=data_transform["train"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                              collate_fn=train_dataset.collate_fn)  # num_workers=nw,

    val_dataset = CLSDataSet(images_path=val_images_path, images_class=val_images_label, transform=data_transform["val"])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            collate_fn=val_dataset.collate_fn)  # num_workers=nw,

    model = AlexNet(num_classes=args.num_classes)
    loss_func = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        loss_func = loss_func.cuda()

    pg = [p for p in model.parameters() if p.requires_grad]   # 是True 则进行梯度更新

    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    tb_writer = SummaryWriter(log_dir=args.output_dir)   # tensorboard --logdir="./output"
    # 计算多长时间记录一下loss lr 准确率的变化
    total_steps = len(train_loader) * args.epochs
    # 希望打印1000次变化  如果总的训练次数小于一千  那间隔就是1   如果总的训练次数大于一千 那就是总的次数/1000
    print_step = 1 if total_steps < 1000 else int(total_steps // 1000)
    train_loss_am = AverageMeter()
    train_acc_am = AverageMeter()
    global_step = 0
    best_acc = None
    for epoch in range(args.epochs):
        # train
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            global_step += 1
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            images, labels = batch    # torch.Size([16, 3, 224, 224])   # torch.Size([16])
            pred = model(images)
            loss = loss_func(pred, labels)
            loss.backward()

            cur_acc = metrics.accuracy_score(labels.cpu().detach().numpy().tolist(),
                                                  torch.max(pred, dim=1)[1].cpu().detach().numpy().tolist())

            print('epoch:{}, step:{}, loss:{:10f}, acc:{:10f}'.format(epoch, step, loss, cur_acc))
            optimizer.step()
            optimizer.zero_grad()

            train_loss_am.update(loss.item())
            train_acc_am.update(cur_acc)
            if global_step % print_step:
                tb_writer.add_scalar('global_step/train_loss', train_loss_am.avg, global_step)
                tb_writer.add_scalar('global_step/train_acc', train_acc_am.avg, global_step)

        scheduler.step()   # 学习率衰减一下
        val_acc = evaluate(model=model, data_loader=val_loader)
        tb_writer.add_scalar('epoch/val_acc', val_acc, epoch)
        tb_writer.add_scalar('epoch/lr', optimizer.param_groups[0]['lr'], epoch)
        print('epoch:{}, val_acc:{:10f}'.format(epoch, val_acc))
        if best_acc is None or val_acc > best_acc:
            model_save_path = os.path.join(args.output_dir, 'best_model.bin')
            torch.save(model.state_dict(), model_save_path)
        model_save_path = os.path.join(args.output_dir, 'epoch{}_model.bin'.format(epoch))
        torch.save(model.state_dict(), model_save_path)