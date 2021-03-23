import glob
import os
import sys

sys.path.append("..")
import utils
from model import *
import torch
import onnx
import torch.onnx

exp_name = "MobileNet_exp1"
which_epoch = 11  # 测试哪个epoch的模型


def pth2onnx2mnn():
    label_path = '../experiment/{}/label.txt'.format(exp_name)
    index2label = utils.read_label(label_path)

    num_class = len(index2label)  # 预测类别数量

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda设备
    net = MobileNet_softmax(num_class).to(device)
    model_dir = os.path.join('../experiment', exp_name, 'checkpoints')
    model_path = os.path.join(model_dir, 'MobileNet_{}.pth'.format(which_epoch))
    net.load_state_dict(torch.load(model_path))  # 加载pytroch的模型
    net.to(device)

    onnx_path = model_path.replace("checkpoints", "onnx").replace('pth', 'onnx')
    mnn_path = model_path.replace('checkpoints', 'mnn').replace('pth', 'mnn')

    onnx_dir = model_dir.replace("checkpoints", "onnx")
    mnn_dir = model_dir.replace("checkpoints", "mnn")
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    if not os.path.exists(mnn_dir):
        os.makedirs(mnn_dir)

    # Create the right input shape (e.g. for an image)
    input = torch.randn(1, 3, 224, 224).to(device)

    input_names = ["image"]
    output_names = ["classifier"]

    # transform model
    torch.onnx.export(net, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names)

    # Load the ONNX model
    model = onnx.load(onnx_path)
    print(model)
    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)

    if not os.path.exists(mnn_path):
        command = './MNNConvert -f ONNX --modelFile {} --MNNModel {} --bizCode biz'.format(
            onnx_path, mnn_path)
        print(command)
        for line in os.popen(command).readlines():
            print(line)

if __name__ == '__main__':
    pth2onnx2mnn()
