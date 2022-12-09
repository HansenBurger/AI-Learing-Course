# 推理和训练(程序)

## 一、Keras手写数字识别

优点:

1. **简易快速的原型设计**
2. 支持CNN和RNN或者二者的结合
3. 无缝CPU和GPU切换

### 1.1 加载训练和检测数据集

初次运行需要下载数据(mnist数据集)

```python
from tensorflow.keras.datasets import mnist
(train_image, train_labels), (test_image, test_labels) = mnist.load_data()
```

训练集和测试集分别有图片和标签

1. 训练集: train_image (60000, 28, 28)
2. 训练标签: train_labels
3. 测试集: test_image (20000, 28, 28)
4. 测试标签: test_labels

### 1.2 搭建神经网路

1.构建空模型(全连接)

```python
from tensorflow.keras import models
network = model.Sequential()
```

2.用layers搭建神经网络层

```python
network.add(layers.Dense(512, activation='relu', input_shape=(28*28)))
```

其中"Dense"用于构建一个数据处理层(参数1:神经元个数，参数2:激活函数，参数3:输入个数)

3.构建输出层

```python
network.add(layers.Dense(10, activation='softmax'))
```

此时使用的激活类为**softmax**。

#### 1) 输出激活 Softmax

1. 二分类，0/1区分
2. 多分类，onehot
3. 真实分类，不可解释的浮点数
4. softmax，**归一化 $[0,1]$**，最终分布结果依据给出的概率

Softmax的公式可以参考:

$$
\begin{equation}
S_i = \frac{e^{Z_i}}{\sum_j{e^{Z_j}}}
\end{equation}
$$

其中，$Z_i$ 表示计算输出的第 $i$ 个神经元，$S_i$ 表示他的输出结果，具体过程可以参考下图:

![softmax_layer](https://www.researchgate.net/publication/333411007/figure/fig9/AS:766785846530048@1559827400305/Example-of-Soft-max-layer.png)

> 综合比较，softmax充分考虑每一个节点的信息，更适合输出的激活

#### 2) 编译

```c++
network.compile(
    optimizer='rmsprop', 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

1. optimizer: 优化
2. loss: 损失函数
3. metrics: 准确度

### 1.3 预处理

处理输入数据用于神经网络训练:

1. reshape，二维转一维，用像素作为输入
2. 对像素值归一化处理(转换数据类型)
3. label转换，把label的值转换为one hot表示(**几类就有几位**)，用于和输出节点对应

对应的程序可以表示为:

```python
from tensorflow.keras.utils import to_categorical
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
```

### 1.4 训练和推理

训练过程:

```python
network.fit(train_imags, train_labels, epochs=5, batch_size=128)
```

推理过程:

```python
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
```

1. 网络结构的变化不影响推理和训练的过程
2. 训练过程不区分领域，实际训练原理和步骤一致

## 二、手推网络架构

### 2.1 网络基本架构

1. 初始化(输入层，中间层，输出层，权重矩阵(根据层数决定))
2. 训练
3. 推理

### 2.2 训练过程

基本的训练过程:

1. 输入训练数据，给出网络结果(**同推理**)
2. 计算误差, 做损失函数反向传播

#### 1) 归一化优化

避免数值小的数据在归一化过程中，权重出现问题，保持归一化的结果在[0.01, 1]之间

> 有些竖直很小，除以255会变0，不利于权重更新

```python
scale_imput = image_arr / 255.0 * 0.99 + 0.01
```

#### 2) 迭代

两个循环，外循环:代，内循环:数据输入(可以分batch)

### 2.3 推理过程

接收输入数据，输出答案

1. 乘加，加权求和
2. 过激活(sigmoid:lambda实现，直接送入加权求和)
3. 在输出层重复此过程

argmax: 输出最大值对应编号(排序过程)

```python
label = numpy.argmax(output)
```

如对于 \[0.02, 0.03, 0,95] 输出的结果为 2
