#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author : LIAOH
@file   : data_loader.py
# @Time : 2020/7/13 20:58
"""

import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_dir, data_path, train=True, need_path=False):

        """
        :param data_dir:    数据集文件夹
        :param data_path:   数据集txt文件路径
        :param train:       训练集 or 测试集
        :param need_path:   是否返回图片的路径

        """

        with open(data_path, 'r', encoding="utf-8") as f:  # 打开数据集txt文件
            paths = []  # 保存图片路径列表
            label = []  # 保存图片标签列表
            for line in f:  # 遍历txt文件的每一行
                line = line.rstrip()  # 去除回车
                words = line.split('\t')  # 拆分路径和标签
                paths.append(data_dir + words[0])  # 更新路径列表，绝对路径转换为相对路径
                label.append(int(words[1]))  # 更新标签列表
        if train:  # 训练集的transform
            self.transform = transforms.Compose([  # 训练集的transform
                transforms.Resize((256, 256)),  # 调整图像大小
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(0.5),  # 50%概率翻转
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # R,G,B每层的归一化用到的均值和方差
            ])
        else:  # 验证集的transform
            self.transform = transforms.Compose([  # 验证集的transform
                transforms.Resize((224, 224)),  # 调整图像大小
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # R,G,B每层的归一化用到的均值和方差
            ])

        self.paths = paths
        self.label = label
        self.need_path = need_path

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB')  # 打开图片，转换为RGB格式
        image = self.transform(image)  # 图片进行transform操作
        if self.need_path:
            return image, self.label[index], self.paths[index]  # 返回图片，标签，图片绝对路径
        else:
            return image, self.label[index]  # 返回图片，标签

    def __len__(self):
        return len(self.paths)


def load_data(data_dir, data_txt, batch_size, train=True, need_path=False, shuffle=False, num_workers=4):
    """data_loader.py

    :param data_dir:    数据集文件夹
    :param data_txt:    数据集标签文件路径
    :param batch_size:  批次大小
    :param train:       训练集 or 测试集
    :param need_path:   是否返回图片的路径
    :param shuffle:     是否打乱
    :returns train_loader, val_loader： 训练集和测试集的DataLoader
    """

    # 创建datasets
    datasets = MyDataset(data_dir, data_txt, train=train, need_path=need_path)

    # 创建DataLoader
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # 返回DataLoader
    return dataloader


def test_dataloader(loader):
    # 查看图片
    count = 0
    for data in loader:
        if count >= 10:  # 只显示10张图片
            break
        images, labels = data  # 数据解包
        image = images[0].numpy()  # 取出该批次数据中的第0个样本图片并转换为ndarray
        img = np.transpose(image, (1, 2, 0))  # 把channel那一维放到最后
        print(img)
        plt.imshow(img)  # 显示图片
        plt.show()
        print(img.shape)  # 打印图片形状
        print(labels[0])  # 打印第0个样本图片的label
        count += 1


def main():
    train_dir = '/data/boyun/train/'
    train_txt = 'data/val.txt'
    batch_size = 32
    train_loader = load_data(train_dir, train_txt, batch_size, train=True)
    test_dataloader(train_loader)


if __name__ == '__main__':
    main()
