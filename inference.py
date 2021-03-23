#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author : LIAOH
@file   : inference.py
# @Time : 2020/8/18 15:26
"""
import os
import torch
import model
from torchvision import transforms
from PIL import Image
from utils import utils

image_dir = '/data/boyun/test/'
exp_name = "MobileNet_exp1"
which_epoch = 11  # 测试哪个epoch的模型
rename = True


def read_label(label_path):
    # 读取标签映射表
    index2label = []
    with open(label_path, encoding='utf-8') as f:
        for line in f:
            index2label.append(line.strip())
    return index2label

def Demo():
    # label_path = './experiment/{}/label.txt'.format(exp_name)
    # model_path = os.path.join(model_dir, 'MobileNet_{}.pth'.format(which_epoch))

    label_path = './experiment/{}/label.txt'.format(exp_name)
    model_path = os.path.join(model_dir, 'MobileNet_{}.pth'.format(which_epoch))

    # 获取标签的映射
    index2label = utils.read_label(label_path)
    num_class = len(index2label)  # 预测类别数量

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda设备
    net = model.MobileNet(num_class).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # 图像transform的操作
    transform = transforms.Compose([  # 验证集的transform
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # R,G,B每层的归一化用到的均值和方差
    ])

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            try:
                
                file_path = os.path.join(root, file)
                image = Image.open(file_path).convert('RGB')  # 打开图片，转换为RGB格式
                image = transform(image)  # 图像transform的操作
            except Exception:
                print("文件 {} 读取失败！！".format(file))
                continue


def inference_():

    for root, dirs, files in os.walk(image_dir):
        i = 0
        for file in files:
            try:
                file_path = os.path.join(root, file)
                image = Image.open(file_path).convert('RGB')  # 打开图片，转换为RGB格式
                image = transform(image)  # 图像transform的操作
            except Exception:
                print("文件 {} 读取失败！！".format(file))
                continue

            image = image.unsqueeze(0).to(device)  # 增加一个batchsize维度

            logist = net(image)  # 获取输出
            prob = torch.softmax(logist, dim=1)
            score = prob.max(dim=1).values.item()  # 获取最高预测score
            pred = torch.argmax(logist, dim=1)

            if pred < 19:
                pred_label = index2label[pred]
            else:
                pred_label = "其他类"

            content = '{}\t{}\t{:.3f}'.format(file, pred_label, score)
            print(content)

            if rename:
                src_file = os.path.join(root, file)
                while True:
                    dst = '{}-{:.3f}-{}.jpg'.format(pred_label, score, i)
                    dst_file = os.path.join(root, dst)
                    i += 1
                    if not os.path.exists(dst_file):
                        break
                os.rename(src_file, dst_file)


if __name__ == '__main__':
    inference()
