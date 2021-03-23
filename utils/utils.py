#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author : LIAOH
@file   : utils.py
# @Time : 2020/8/18 10:04
"""

import torch.nn.functional as F
import os
from collections import Iterable
from model import *
import torch
import torch.onnx
import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.quantum.circuitplot import matplotlib


def read_label(label_path):
    # 读取标签映射表
    index2label = []
    with open(label_path, encoding='utf-8') as f:
        for line in f:
            index2label.append(line.strip())
    return index2label


def confusion_matrix(preds, labels, conf_matrix):
    # 混淆矩阵更新
    # preds = torch.argmax(preds, 1)  # 获取预测结果
    for p, t in zip(preds, labels):  # 更新混淆矩阵
        if p > 19:
            p = 19
        conf_matrix[t, p] += 1  # t:label    p:preds
    return conf_matrix


def plot_Matrix(cm, classes, result_path, file_name, title=None, cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    :param cm: 混淆矩阵
    :param classes: 类名
    :param result_path: 混淆图片保存路径
    :param title: 图片标题
    :param cmap:
    :return:
    """
    plt.rc('font', family='Times New Roman', size='8')  # 设置字体样式、大小
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm.shape)
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    matplotlib.rcParams['font.family'] = 'SimHei'  # 修改了全局变量，设置为黑体

    matplotlib.rcParams['font.size'] = 4
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)  # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    plt.savefig(os.path.join(result_path, file_name), dpi=300)
    plt.show()


def get_class(txt_path):
    # 获取标签映射表
    class_name = []
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            line = line.rsplit()[0]
            class_name.append(line)
    return class_name


def get_alpha(txt, num):
    alpha_cls = np.array([0 for _ in range(num)])
    with open(txt, 'r', encoding="utf-8") as f:  # 打开数据集txt文件
        for line in f:  # 遍历txt文件的每一行
            line = line.rstrip()  # 去除回车
            words = line.split('\t')  # 拆分路径和标签
            label = int(words[1])
            alpha_cls[label] += 1
    alpha_cls = sum(alpha_cls) / alpha_cls

    return alpha_cls


def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)


def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_binary_tensor(tensor, boundary=19):
    zero = torch.zeros_like(tensor)
    one = torch.ones_like(tensor)
    return torch.where(tensor >= boundary, one, zero)


class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)

            logit = logit.permute(0, 2, 1).contiguous()

            logit = logit.view(-1, logit.size(-1))

        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            # one_hot_key限制在[smooth, 1.0 - smooth]区间内
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
