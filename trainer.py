import time
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from earlystopping import EarlyStopping
from utils.readData import read_dataset
from torchvision.models import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(net, loss, train_dataloader, valid_dataloader, device, num_epoch, lr, optim='adam',scheduler_type='Cosine', init = True):
    def init_xavier(m):
        #if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
    if init:
        net.apply(init_xavier)
        
    print('training on:', device)
    net.to(device)
    # 优化器选择
    if optim == 'sgd':
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=0)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=0)
    elif optim == 'adamW':
        optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=0)
    # elif optim == 'ranger':
    #     optimizer = Ranger((param for param in net.parameters() if param.requires_grad), lr=lr,
    #                        weight_decay=0)
    # if scheduler_type == 'Cosine':
    #     lr_min = 0
    #     scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)

    # 用来保存每个epoch的Loss和acc以便最后画图
    train_losses = []
    train_acces = []
    eval_losses = []
    eval_acces = []
    best_acc = 0.0
    # path = ".\\checkpoint\\" + net.__class__.__name__ + '-latest'+ '.pth'
    early_stopping = EarlyStopping(verbose=True)
    # 训练
    for epoch in range(num_epoch):
        print("————————————第 {} 轮训练开始————————————".format(epoch + 1))
        time.sleep(1)
        # 训练开始
        net.train()
        train_acc = 0
        train_loss = 0
        for batch in tqdm(train_dataloader, desc='train'):
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = net(imgs)
            Loss = loss(output, targets)
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item()
            train_acc += num_correct / imgs.shape[0]
            train_loss += Loss.item()

        # scheduler.step()
        train_acces.append(train_acc / len(train_dataloader))
        train_losses.append(train_loss / len(train_dataloader))

        # 测试步骤开始
        net.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for imgs, targets in tqdm(valid_dataloader,desc='valid'):
                imgs = imgs.to(device)
                targets = targets.to(device)
                output = net(imgs)
                Loss = loss(output, targets)
                _, pred = output.max(1)
                num_correct = (pred == targets).sum().item()
                eval_loss += Loss.item()
                eval_acc += num_correct / imgs.shape[0]
            if eval_acc > best_acc:
                best_acc = eval_acc
                best_path = '.\\checkpoint\\' + net.__class__.__name__ + '-best' + '.pth'
                torch.save(net.state_dict(), best_path)

            eval_losses.append(eval_loss / len(valid_dataloader))
            eval_acces.append(eval_acc / len(valid_dataloader))
            print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'.format(epoch+1, train_loss / len(train_dataloader), train_acc / len(train_dataloader),eval_loss / len(valid_dataloader), eval_acc / len(valid_dataloader)))
            early_stopping(eval_loss, net)
            if early_stopping.early_stop:
                print("Early stopping!")
                break  # 跳出迭代，结束训练

    return train_losses, train_acces, eval_losses, eval_acces

def show_acces(train_losses, train_acces, valid_losses ,valid_acces, num_epoch):#对准确率和loss画图显得直观
    plt.plot(1 + np.arange(len(train_losses)), train_losses, linewidth=1.5, linestyle='dashed', label='train_losses')
    plt.plot(1 + np.arange(len(train_acces)), train_acces, linewidth=1.5, linestyle='dashed', label='train_acces')
    plt.plot(1 + np.arange(len(valid_losses)), valid_losses, linewidth=1.5, linestyle='dashed', label='valid_loss')
    plt.plot(1 + np.arange(len(valid_acces)), valid_acces, linewidth=1.5, linestyle='dashed', label='valid_acces')
    plt.grid()
    plt.xlabel('epoch')
    plt.xticks(range(1, 1 + num_epoch, 1))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    net = vgg16(num_classes=10, pretrained=False)
    train_loader, valid_loader, test_loader = read_dataset(batch_size=64, pic_path='dataset')
    loss = nn.CrossEntropyLoss()
    lr = 1e-4
    num_epoch = 100
    train_losses, train_acces, eval_losses, eval_acces = train(net, loss, train_loader, test_loader, device, num_epoch, lr)
    # 画图
    show_acces(train_losses, train_acces,eval_losses ,eval_acces, num_epoch=1)
