# reference from 
# https://onnxruntime.ai/docs/get-started/with-python.html

# PyTorch 
# export the model using `torch.onnx.export`
torch.onnx.export(model,                                # model being run
                  torch.randn(1, 28, 28).to(device),    # model input (or a tuple for multiple inputs)
                  "fashion_mnist_model.onnx",           # where to save the model (can be a file or file-like object)
                  input_names = ['input'],              # the model's input names
                  output_names = ['output'])            # the model's output names

# load the onnx model with `onnx.load`
import onnx
onnx_model = onnx.load("fashion_mnist_model.onnx")
onnx.checker.check_model(onnx_model)

# create inference session usning `ort.InferenceSession`
import onnxruntime as ort
import numpy as np
x, y = test_data[0][0], test_data[0][1]
ort_sess = ort.InferenceSession('fashion_mnist_model.onnx')
outputs = ort_sess.run(None, {'input': x.numpy()})

# Print Result 
predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
print(f'Predicted: "{predicted}", Actual: "{actual}"')

# PyTorch NLP
# Process text and create the sample data input and offsets(偏移量) for export.
import torch
text = "Text from the news article"
text = torch.tensor(text_pipeline(text))
offsets = torch.tensor([0])

# export model
# Export the model
# export函数的参数要好好研究
torch.onnx.export(model,                     # model being run
                (text, offsets),           # model input (or a tuple for multiple inputs)
                "ag_news_model.onnx",      # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=10,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization（是否执行常量折叠进行优化——常量折叠是一种编译器优化技术）
                input_names = ['input', 'offsets'],   # the model's input names
                output_names = ['output'], # the model's output names
                dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes(可变长度轴)  作用：将第一维指定为动态
                              'output' : {0 : 'batch_size'}})

# load the model using `onnx.load`
import onnx
onnx_model = onnx.load("ag_news_model.onnx")
onnx.checker.check_model(onnx_model)

# create inference session with `ort.inference`
import onnxruntime as ort
import numpy as np
ort_sess = ort.InferenceSession('ag_news_model.onnx')
outputs = ort_sess.run(None, {'input': text.numpy(),
                            'offsets':  torch.tensor([0]).numpy()})  # sess.run([output_name], {input_name: x})
# Print Result
# 输出数据处理,视不同的项目而定
result = outputs[0].argmax(axis=1)+1
print("This is a %s news" %ag_news_label[result[0]])