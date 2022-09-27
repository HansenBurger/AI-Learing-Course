# 框架

## 一、用keras实现一个简单的神经网络

### 1.1 keras

优点:
    1. 简易快速的原型设计
    2. 支持CNN和RNN或者二者的结合
    3. 无缝CPU和GPU切换

1. 加载数据

加载训练和检测数据集

```python
from tensorflow.keras.datasets import mnist
(train_image, train_labels), (test_image, test_labels) = mnist.load_data()
```

dense first param 隐藏层的数量(输出shape)

不同的隐藏层参数不同

构建神经网络没有标准答案

激活类: softmax

### 1.2 softmax

用于**多分类**过程, 映射到[0, 1]之间, 看成概率, 来进行多分类

$ S_i = \frac{e^{V_i}}{\SUM{e^{V_i}}}$

网络构建结束就要编译

optimizer: 优化
loss: 损失函数
metrics: 准确度

预处理:

二维 --> 一维
归一化: [0, 255] --> [0, 1]
label转换: 值 --> one hot (to_categorical) 为了输出和标签对应

输入神经网络:

```python
network.fit(train_imags, train_labels, epochs=5, batch_size=128)
```

训练结果, 训练和验证准确度以及loss不会一致

网络结构的变化不影响推理和训练的过程

## 二、网络搭建

### 2.1 网络基本架构

1. 初始化

2. 训练

3. 查询

### 2.3 训练过程

1. 输入训练数据
2. 计算误差, 做损失函数反向传播

避免数值小的数据在归一化过程中, 权重出现问题, [0.01, 1]

argmax -> 输出最大值对应编号
