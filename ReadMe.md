# 一 、环境配置

项目需要以下环境

- Python-3.7
- torch-1.3.1
- torchvision-0.4.2
- onnx
- numpy
- matplotlib
- Pillow
- sympy

可以通过以下命令安装对应依赖包：

```
pip install -r requirement.txt
```

# 二、数据集准备

## 1. 数据集文件结构

```
├─狗
│  ├─多条狗
│  │      20200810-04-1.jpg
│  │      ...
│  ...
│  └─正侧趴
│          20200810-4-102.jpg
│          ...
├─蓝天
│      20200810-6-127.jpg
│      ...
...
```

上面是数据集文件夹的结构示意图，每个类别中可能包含子文件夹或者只包含图片。训练集和测试集的文件结构保持一致。

## 2. 数据处理

将训练集和测试集按照上述文件结构分别放入不同文件夹中，修改`prepare_data.py`中以下信息：

```python
labels = ['人像', '夜景', '婴儿', '室内', '投影幕', '文本文档', '日出日落', '束光灯', '海滩', '焰火', '狗', '猫', '绿植',
          '绿草地', '美食', '蓝天', '逆光', '雪景', '风景']			# 数据集文件夹名

num_example = 5000  # 每个类别最大样本数量

exp_name = "MobileNet_exp1"  # 实验名称

train_dir = '/data/train/'	# 训练集文件夹
val_dir = '/data/val/'		# 测试集文件夹
```

修改完成后运行`prepare_data.py`即可在`/experiment/exp_name/`文件夹下生成`train.txt`、`val.txt`和`label.txt`

# 三、训练

修改`train.py`中以下参数

```python
exp_name = "MobileNet_exp1"			# 实验名称
train_dir = '/data/train/'    # 训练集文件夹
val_dir = '/data/val/'        # 测试集文件夹

lr = 0.0001  # 学习率
batch_size = 512  # 批处理大小
num_classes = 19  # 预测类别数量，请保证和label.txt中的类别数量相同
total_epoch = 30  # 遍历数据集次数

is_test = Fasle  # 训练过程中是否进行测试
continue_train = Fasle  # 是否继续训练
which_epoch = 11  # 如果继续训练，应该加载哪个epoch
max_acc = [95.255]  # 模型在验证集上的最高精确度, 继续训练才会用到,修改为之前训练的最高精确度，模型只保留大于90%以上的最佳模型。
```

运行`train.py`即可开始训练，训练过程中的模型和日志文件会保存到`/experiment/exp_name/checkpoints`

# 四、测试

`inference.py`用于预测某个文件夹中全部图片的结果，修改其中以下参数

```python
image_dir = '/data/test/'	# 测试图片的文件夹路径
exp_name = "MobileNet_exp1"	# 测试哪次实验的模型
which_epoch = 11  	# 测试哪个epoch的模型
rename = True	# 是否利用预测结果修改预测图片文件名
```

运行`inference.py`即可预测`image_dir`下图片的预测结果，如果选择重命名，对应的图片将会利用预测结果进行重命名。

# 五、评估

`evaluate.py`是用于指标评估的脚本，可以测试模型在指定数据集的预测精确度和相关混淆矩阵。

参看2.2中`val.txt`的生成，同样的方法生成`test.txt`文件，然后修改以下参数

```python
exp_name = "MobileNet_exp1"	# 评估哪次实验的模型
test_dir = '/data/test/'	# 测试集路径
which_epoch = 11  # 测试哪个epoch的模型
batch_size = 8
```

运行`inference.py`后，控制台会打印每个类别的精确度和平均精确度，`/experiment/exp_name/`下会生成对应的混淆矩阵图片。

# 六、转换

转换目前只能在linux平台进行，`utils/convert.py`的可以将pytroch的模型转换为onnx和mnn模型，需要修改以下参数：

```python
exp_name = "MobileNet_exp1"	# 转换哪次实验的模型
which_epoch = 11  # 转换哪个epoch的模型
```

将目录切换到`utils`下，再运行`convert.py`即可在`/experiment/exp_name/onnx`和`/experiment/exp_name/mnn`文件夹下生成对应的onnx和mnn模型文件。