#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author : LIAOH
@file   : main.py
# @Time : 2020/7/13 20:33
"""
import argparse
import os

import model
import torch.optim as optim
from data_loader import *
from utils import Logger, utils

exp_name = "MobileNet_exp1"
train_dir = '/data/boyun/train/'    # 训练集文件夹
val_dir = '/data/boyun/val/'        # 测试集文件夹


lr = 0.0001  # 学习率
batch_size = 8  # 批处理大小
num_classes = 19  # 预测类别数量

total_epoch = 30  # 遍历数据集次数

is_test = False  # 训练过程中是否进行测试
continue_train = False  # 是否继续训练
which_epoch = 11  # 如果继续训练，应该加载哪个epoch
max_acc = [87.255]  # 模型在验证集上的最高精确度, 继续训练才会用到,修改为之前训练的最高精确度

step_size = 5  # 多少epoch衰减gamma倍学习率
gamma = 0.8  # 每次衰减倍速

train_txt = './experiment/{}/train.txt'.format(exp_name)  # 训练集标签文件
val_txt = './experiment/{}/val.txt'.format(exp_name)  # 训练集标签文件
label_path = './experiment/{}/label.txt'.format(exp_name)

assert num_classes == len(utils.read_label(label_path))

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=exp_name, help='实验名称, 决定在哪里存储样本和模型')
    parser.add_argument('--train_txt', type=str, default=train_txt, help='训练集标签文件')
    parser.add_argument('--train_dir', type=str, default=train_dir, help='数据集文件夹')
    parser.add_argument('--val_txt', type=str, default=val_txt, help='验证集标签文件')
    parser.add_argument('--val_dir', type=str, default=val_dir)
    parser.add_argument('--lr', type=float, default=lr, help='学习率')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='输入批次大小')
    parser.add_argument('--num_classes', type=int, default=num_classes, help='预测类别数量')
    parser.add_argument('--total_epoch', type=int, default=total_epoch, help='训练的epoch数')
    parser.add_argument('--is_test', action='store_true', default=is_test, help='继续培训：加载最新模型')
    parser.add_argument('--continue_train', action='store_true', default=continue_train, help='继续培训：加载最新模型')
    parser.add_argument('--which_epoch', type=int, default=which_epoch, help='如果继续训练，应该加载哪个epoch')

    opt = parser.parse_args()

    if not continue_train:
        opt.which_epoch = 0

    opt.model_dir = os.path.join('./experiment', opt.exp_name, 'checkpoints')
    if not os.path.exists(opt.model_dir):
        utils.mkdirs(opt.model_dir)
    return opt


def load_model(opt, log, device):
    net = model.MobileNet(opt.num_classes).to(device)

    if opt.continue_train:
        model_path = os.path.join(opt.model_dir, 'MobileNet_{}.pth'.format(opt.which_epoch))
        log.info("restore model from:MobileNet_{}.pth".format(opt.which_epoch))
        net.load_state_dict(torch.load(model_path))

    log.info("Model loaded successfully!!")

    return net


def main():
    def train():

        log.info('\nEpoch: %d' % epoch)

        # 模型用于训练
        net.train()
        sum_loss = 0.0  # 总loss
        total = 0.0  # 预测的样本数量
        correct = 0.0  # 预测正确的图片数量
        length = len(train_loader)

        for i, data in enumerate(train_loader):  # 遍历训练集的全部批次

            inputs, labels = data  # 数据解包
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零
            outputs = net(inputs)  # 前向传播

            loss = criterion(outputs, labels)  # 定义损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            sum_loss += loss.item()  # 更新总loss
            total += labels.size(0)  # 更新预测的样本数量

            predicted = torch.argmax(outputs, dim=1)
            correct += torch.eq(predicted, labels).cpu().sum().float().item()  # 更新预测正确的图片数量
            if i % 10 == 0:
                # 打印信息
                log.info('[epoch:%d, iter:%d/%d] Loss: %.05f | Acc: %.3f%% '
                         % (epoch, (i + 1), length, sum_loss / (i + 1), 100. * correct / total))

    def test():
        # 保存模型
        model_path = os.path.join(opt.model_dir, 'MobileNet_{}.pth'.format(epoch))
        torch.save(net.state_dict(), model_path)
        log.info('save model to :{}'.format(model_path))
        # 每训练完一个epoch测试一下准确率
        log.info("Waiting Test!")
        net.eval()  # 模型用于评估
        with torch.no_grad():
            total = 0.0  # 预测的样本数量
            correct = 0.0  # 预测正确的图片数量

            for i, data in enumerate(val_loader):  # 遍历验证集集的全部批次
                inputs, labels = data  # 数据解包
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)  # 获取输出
                predicted = torch.argmax(outputs, dim=1)

                total += labels.size(0)  # 更新预测的样本数量

                correct += torch.eq(predicted, labels).cpu().sum().float().item()  # 更新预测正确的图片数量
            val_acc = 100. * correct / total
            log.info('测试分类准确率为：%.3f%% ' % (100. * correct / total))

        if val_acc > max_acc[0] and val_acc > 90.:
            max_acc[0] = val_acc  # 只有准确度达到了90%，并且精确度提升了才更新最佳模型
            best_model_path = os.path.join(opt.model_dir, 'MobileNet_best_{}.pth'.format(epoch))
            torch.save(net.state_dict(), best_model_path)
            log.info('save best model to :{}'.format(best_model_path))

        return max_acc

    # 参数解析
    opt = get_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 或者 CPU

    # 加载数据集
    train_loader = load_data(opt.train_dir, opt.train_txt, opt.batch_size, train=True, shuffle=True)
    val_loader = load_data(opt.val_dir, opt.val_txt, opt.batch_size, train=False, shuffle=False)

    log = Logger.Log(os.path.join(opt.model_dir, "train.log"))  # 日志文件

    # 加载模型
    net = load_model(opt, log, device)

    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)  # 优化器
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.8)
    # criterion = nn.CrossEntropyLoss().to(device=device)  # 交叉熵损失函数

    alpha_cls = utils.get_alpha(opt.train_txt, opt.num_classes)  # Focal loss 参数alpha
    criterion = utils.MultiFocalLoss(num_class=opt.num_classes, alpha=alpha_cls)  # Focal loss

    log.info("Start Training!")

    for epoch in range(opt.which_epoch + 1, opt.total_epoch + 1):
        train()
        if opt.is_test:
            test()
        else:
            model_path = os.path.join(opt.model_dir, 'MobileNet_{}.pth'.format(epoch))
            torch.save(net.state_dict(), model_path)
            log.info('save model to :{}'.format(model_path))
        StepLR.step()


if __name__ == '__main__':
    main()
