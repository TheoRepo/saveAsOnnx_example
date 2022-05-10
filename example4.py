# reference from
# https://pytorch.org/docs/stable/onnx.html

import torch
import torchvision

# 模型的输入
# 10 代表batchsize
# 3 代表 channel
# 224 代表 length
# 224 代表 width
dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
# 加载预训练模型
model = torchvision.models.alexnet(pretrained=True).cuda()

# 输入输出的名字
# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
# ['actual_input_1', 'learned_0', 'learned_1', 'learned_2', 'learned_3', 'learned_4', 'learned_5', 'learned_6', 'learned_7', 'learned_8', 'learned_9', 'learned_10', 'learned_11', 'learned_12', 'learned_13', 'learned_14', 'learned_15']
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
