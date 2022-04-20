# 代码逻辑
# 加载图像分类模型
# 并根据包含的测试数据确认其准确性。

# 写代码的思路
# 分析代码结构
# 写代码，人的脑子也讲究注意力机制，把注意力放在代码块的功能上，代码块的功能串联起来，就会理解一个脚本的代码逻辑
# 反之，把注意力放在打字的正确性，按键在哪里？人只会犯困想睡觉。因为思路被一次又一次地打断了。
# 梳理完代码逻辑，接下来注意力要放在，重要函数的用法，掌握重要函数的用法，我会就掌握了改代码的能力，面对不同业务需求，举一反三

import numpy as np
import onnxruntime
import onnx
from onnx import numpy_helper
import urllib.request
import json
import time
import datetime
from download import download

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

onnx_model_url = "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz"
imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

# 从 ONNX 模型动物园中检索我们的模型
download(onnx_model_url, savepath='./')
download(imagenet_labels_url, savepath='./')

# 加载样本输入和输出
test_data_dir = 'resnet50v2/test_data_set'
test_data_num = 3

import glob
import os

# 加载数据
# 加载输入
inputs = []
for i in range(test_data_num):
    input_file = os.path.join(test_data_dir + '_{}'.format(i), 'input_0.pb')
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
        inputs.append(numpy_helper.to_array(tensor))

print('Loaded {} inputs successfully.'.format(test_data_num))
print(np.array(inputs).shape) 
# 输入数据的结构(3, 1, 3, 224, 224) 
# (3)表示3个数据文件，
# (1)表示第一张图，
# (3, 224, 224)是ResNEt50网络的输入（C,H,W） (channel, height, width)
print(inputs[0])

# 加载推理输出
ref_outputs = []
for i in range(test_data_num):
    output_file = os.path.join(test_data_dir + '_{}'.format(i), 'output_0.pb')
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
        tensor.ParseFromString(f.read())    
        ref_outputs.append(numpy_helper.to_array(tensor))        
print('Loaded {} reference outputs successfully.'.format(test_data_num))
print(np.array(ref_outputs).shape) 
# 输出数据的结构(3, 1, 1000) 
# (3)表示3个数据文件，
# (1)表示第一张图，
# (1000)表示1000个分类


# 使用ONNX Runtime推理
# 在后端运行模型
session = onnxruntime.InferenceSession('resnet50v2/resnet50v2.onnx', None)

# 获取模型第一个输入的名称
input_name = session.get_inputs()[0].name
print('Input Name:', input_name)

# 运用推理用时
starttime = datetime.datetime.now()
outputs = [session.run([], {input_name: inputs[i]})[0] for i in range(test_data_num)]
endtime = datetime.datetime.now()
print((endtime - starttime).seconds)

print('Predicted {} results.'.format(len(outputs)))
# 将结果与最多 4 位小数的参考输出进行比较
for ref_o, o in zip(ref_outputs, outputs):
    np.testing.assert_almost_equal(ref_o, o, 4)
print('ONNX Runtime outputs are similar to reference outputs!')

# 重要函数
# onnxruntime.InferenceSession

# np.testing.assert_almost_equal
# 如果两个项目不等于期望值，则引发 AssertionError精确。