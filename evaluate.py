#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author : LIAOH
@file   : evaluate.py
# @Time : 2020/7/14 19:06
"""
import argparse
import os
from data_loader import *
from utils import Logger
from utils import utils
import torch
import model
from utils import utils

exp_name = "MobileNet_exp1"
test_dir = '/data/boyun/test/'
which_epoch = 11  # 测试哪个epoch的模型
batch_size = 8


def evaluate():
    # 获取标签的映射

    label_path = './experiment/{}/label.txt'.format(exp_name)
    test_txt = './experiment/{}/train.txt'.format(exp_name)  # 训练集标签文件
    model_dir = os.path.join('./experiment', exp_name, 'checkpoints')  # 模型文件夹
    model_path = os.path.join(model_dir, 'MobileNet_{}.pth'.format(which_epoch))  # 模型路径
    log = Logger.Log(os.path.join(model_dir, "val.log"))  # 日志文件

    index2label = utils.read_label(label_path)
    num_class = len(index2label)  # 预测类别数量

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda设备
    net = model.MobileNet(num_class).to(device)
    log.info("restore model from:MobileNet_{}.pth".format(which_epoch))

    net.load_state_dict(torch.load(model_path))
    net.to(device)
    net.eval()

    # 获取数据
    val_loader = load_data(test_dir, test_txt, batch_size, train=False, shuffle=False)

    log.info("Start Evaluate!")

    with torch.no_grad():
        # 批次数量
        n = len(val_loader)
        # 每个类中预测正确的样本数量，由于后面计算精确度要用的除法，因此初始化为浮点数的 0.
        correct = list(0. for _ in range(num_class))
        # 每个类中预测的样本总数
        total = list(0. for _ in range(num_class))
        # 初始化混淆矩阵
        conf_matrix = torch.zeros(num_class, num_class)

        # 遍历全部批次的数据
        for i, data in enumerate(val_loader):

            log.info("[iter:{}/{}]".format(i, n))
            # 准备数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 获取输出
            outputs = net(inputs)

            predicted = torch.argmax(outputs, dim=1)

            # 更新混淆矩阵
            conf_matrix = utils.confusion_matrix(predicted, labels, conf_matrix)

            # 遍历该批次全部样本
            for idx in range(len(labels)):
                gt = labels[idx]  # 获取当前样本真实标签的序号
                pre = predicted[idx]  # 获取当前样本预测的序号
                if pre > 19:
                    pre = 19
                if gt == pre:  # 当前样本预测正确
                    correct[gt] += 1  # 更新对应真实标签的类别预测正确样本数量
                total[gt] += 1  # 更新对应真实标签的类别用于预测的样本数量

        # 遍历每一个类别
        for acc_idx in range(num_class):
            try:
                # 单类别准确率 = 对应类别预测正确的样本数量/对应类别预测样本总数
                acc = correct[acc_idx] / total[acc_idx]
            except:
                acc = 0
            finally:
                log.info('class:%s\tacc:%f\t' % (index2label[acc_idx], acc))
        # 平均准确率 = 预测正确的样本数量/预测样本总数
        acc_str = 'Accuracy: %f' % (sum(correct) / sum(total))
        log.info(acc_str)
        # 获取label的名称
        classes = utils.get_class(label_path)
        # 绘制混淆矩阵
        result_dir = os.path.join('./experiment', exp_name)  # 模型文件夹
        cm_file = "{}_混淆矩阵.jpg".format(which_epoch)
        utils.plot_Matrix(conf_matrix.numpy(), classes, result_dir, cm_file)


if __name__ == '__main__':
    evaluate()
