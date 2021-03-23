#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author : LIAOH
@file   : model.py
# @Time : 2020/7/13 21:04
"""
import torch
import torch.nn as nn
from torchvision import models


class MobileNet(nn.Module):
    def __init__(self, num_classes=19):
        super(MobileNet, self).__init__()
        net = models.mobilenet_v2(pretrained=True)
        self.features = net.features
        self.classifier = nn.Sequential(
            nn.Linear(1280, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        logit = self.classifier(x)
        return logit

class MobileNet_softmax(nn.Module):
    # MobileNet v2网络结构,加上softmax，省去转换的MNN模型的softmax操作
    def __init__(self, num_classes=19):
        super(MobileNet_softmax, self).__init__()
        net = models.mobilenet_v2(pretrained=True)
        self.features = net.features
        self.classifier = nn.Sequential(
            nn.Linear(1280, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # features输出拉直
        logits = self.classifier(x)
        pro = torch.softmax(logits, dim=1)
        return pro