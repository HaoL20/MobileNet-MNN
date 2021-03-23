# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author : LIAOH
@file   : prepare_one_class_data.py
# @Time : 2020/8/12 9:35
"""
import os
import random
import shutil

labels = ['人像', '夜景', '婴儿', '室内', '投影幕', '文本文档', '日出日落', '束光灯', '海滩', '焰火', '狗', '猫', '绿植',
          '绿草地', '美食', '蓝天', '逆光', '雪景', '风景']  # 数据集文件夹名

# labels = ['人像', '夜景', '婴儿', '室内', '投影幕', '文本文档', '日出日落', '束光灯', '海滩', '焰火', '狗', '猫', '绿植',
#           '绿草地', '美食', '蓝天', '逆光', '雪景', '风景',
#           '无关类/保温杯', '无关类/写字板', '无关类/垃圾桶', '无关类/手', '无关类/打印机', '无关类/扫把', '无关类/插座',
#           '无关类/模糊', '无关类/灯笼', '无关类/矿泉水瓶', '无关类/磨砂玻璃', '无关类/福', '无关类/红砖墙', '无关类/纯白',
#           '无关类/纸巾', '无关类/背包', '无关类/花盆', '无关类/裤子', '无关类/车', '无关类/键盘', '无关类/鼠标']

num_example = 10  # 每个类别最大样本数量（无关类除外）

exp_name = "MobileNet_exp1"  # 实验名称
train_dir = '/data/boyun/train/'    # 训练集文件夹
val_dir = '/data/boyun/val/'        # 测试集文件夹

exp_dir = './experiment/' + exp_name
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
train_txt = './experiment/{}/train.txt'.format(exp_name)  # 训练集标签文件
val_txt = './experiment/{}/val.txt'.format(exp_name)  # 训练集标签文件
label_path = './experiment/{}/label.txt'.format(exp_name)


def get_list():
    train_files = []  # 训练集标签列表
    val_files = []  # 验证集标签列表
    counts = []  # 每个类别的样本数量

    for idx, label in enumerate(labels):

        count = 0
        cur_train_dir = os.path.join(train_dir, label)  # 当前训练集遍历 label 文件夹目录
        cur_val_dir = os.path.join(train_dir, label)  # 当前验证集遍历 label 文件夹目录
        cur_train_files = []
        cur_val_files = []

        ################# 训练集 #################
        for parent, dirs, files in os.walk(cur_train_dir):
            if not files:  # 跳过没有无图片文件的parent文件夹
                continue
            relative_path = parent.replace(train_dir, '')  # 获取parent文件夹在dir文件夹下的相对路径
            cur_files = [[os.path.join(relative_path, file), idx] for file in files]
            count += len(cur_files)
            if "蛋糕面包" in parent or "水果果盘" in parent:
                cur_train_files.extend(cur_files[:300])
            else:
                cur_train_files.extend(cur_files)
        random.shuffle(cur_train_files)
        # 大于5000张的正常类别，随机挑选5000张
        if len(cur_train_files) > num_example and idx < 19:
            train_files.extend(cur_train_files[:num_example])
            counts.append(num_example)
        # 小于5000张的正常类别，补全5000张
        elif idx < 19:
            l = len(cur_train_files)
            if l == 0:
                counts.append(0)
                continue
            n = int(num_example / l)
            temp = cur_train_files * n + [cur_train_files[i] for i in random.sample(range(0, l), num_example - n * l)]
            train_files.extend(temp)
            counts.append(num_example)
        # 无关类
        else:
            train_files.extend(cur_train_files)
            counts.append(len(cur_train_files))

        ################# 测试集 #################
        for parent, dirs, files in os.walk(cur_val_dir):
            if not files:  # 跳过没有无图片文件的parent文件夹
                continue
            relative_path = parent.replace(train_dir, '')  # 获取parent文件夹在dir文件夹下的相对路径
            cur_files = [[os.path.join(relative_path, file), idx] for file in files]
            count += len(cur_files)
            cur_val_files.extend(cur_files)
        val_files.extend(cur_val_files)
    print(counts)

    return train_files, val_files, labels


def save_train_val_txt(content, data_path):
    # 保存数据集标签列表
    with open(data_path, 'w', encoding='utf-8') as f:
        for path, label in content:
            # 每一行写入： 文件路径 \t 标签编号 \n
            content_str = (path + '\t' + str(label) + '\n').replace("\\", '/')
            f.write(content_str)


def save_label(lables, path):
    # 保存标签映射表
    with open(path, 'w', encoding='utf-8') as f:
        for label in lables:
            content_str = label + '\n'
            f.write(content_str)


def main():
    train_files, val_files, labels = get_list()
    save_train_val_txt(train_files, train_txt)
    save_train_val_txt(val_files, val_txt)
    save_label(labels, label_path)


if __name__ == '__main__':
    main()
