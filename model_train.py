import copy
import time

import pandas as pd

import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
import torch.nn as nn

from torchvision.datasets import FashionMNIST
from torchvision import transforms
from model import vgg16


def train_val_data_process():
    train_dataset = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                              download=True)

    # 数据划分
    train_data, val_data = Data.random_split(train_dataset, [round(0.8*len(train_dataset)), round(0.2*len(train_dataset))])

    train_data = Data.DataLoader(dataset= train_data,
                                   batch_size= 32,
                                   shuffle= True,
                                   num_workers= 0)

    val_data = Data.DataLoader(dataset= val_data,
                                   batch_size= 32,
                                   shuffle= True,
                                   num_workers= 0)

    return train_data, val_data

def train_model_process(model, train_data, val_data, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # 梯度下降优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始参数
    # 最高准确度
    best_acc = 0.0
    # 训练损失列表
    train_loss_all = []
    # 验证损失列表
    val_loss_all = []
    # 训练准确度列表
    train_acc_all = []
    # 验证准确度列表
    val_acc_all = []

    # 当前时间
    since = time.time()
    print(f'开始训练时间：{time.ctime()}')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        train_batch_size = len(train_data)
        val_batch_size = len(val_data)

        train_loss = 0.0
        train_correct = 0.0
        val_loss = 0.0
        val_correct = 0.0
        train_num = 0.0
        val_num = 0.0
        batch_start_time = time.time()

        for step, (b_x, b_y) in enumerate(train_data):
            b_x, b_y = b_x.to(device), b_y.to(device)
            model.train()
            output = model(b_x)
            pre_label = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_correct += torch.sum(pre_label == b_y.data)

            train_num += b_x.size(0)

            if step % 50 == 0:
                batch_time = time.time() - batch_start_time
                print(f'Train Batch:{step+1}/{train_batch_size} in Epoch {epoch+1}, Time use: {batch_time:.2f}s')
                batch_start_time = time.time()

        for step, (b_x, b_y) in enumerate(val_data):
            if step % 50 == 0:
                print(f'Val Batch:{step}/{val_batch_size} in Epoch {epoch+1}')
            b_x, b_y = b_x.to(device), b_y.to(device)
            model.eval()
            # with torch.no_grad():
            output = model(b_x)
            pre_label = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)
            val_correct += torch.sum(pre_label == b_y.data)

            val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_correct.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_correct.double().item() / val_num)

        print(f'{epoch + 1} train_loss:{train_loss_all[-1]:.4f} train_acc: {train_acc_all[-1]:.4f}')
        print(f'{epoch + 1} val_loss:{val_loss_all[-1]:.4f} val_acc: {val_acc_all[-1]:.4f}')

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print(f'训练和验证耗费的时间: {time_use // 60:.0f}m {time_use % 60:.0f}s')

    torch.save(best_model_wts, 'best_model.pth')

    train_process = pd.DataFrame(data={'epoch': range(num_epochs),
                                       'train_loss_all': train_loss_all,
                                       'val_loss_all': val_loss_all,
                                       'train_acc_all': train_acc_all,
                                       'val_acc_all': val_acc_all})

    return train_process


def matplot_acc_loss(process):
    plt.figure(figsize=(12, 4))
    # 一行两列的第1张图
    plt.subplot(1, 2, 1)
    plt.plot(process['epoch'] + 1, process.train_loss_all, 'ro-', label='train_loss')
    plt.plot(process['epoch'] + 1, process.val_loss_all, 'bs-', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    # 一行两列的第2张图
    plt.subplot(1, 2, 2)
    plt.plot(process['epoch'] + 1, process.train_acc_all, 'ro-', label='train_acc')
    plt.plot(process['epoch'] + 1, process.val_acc_all, 'bs-', label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    vgg16 = vgg16()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(vgg16, train_dataloader, val_dataloader, 20)
    matplot_acc_loss(train_process)
