# 深度学习开源框架

## 一、框架核心组件

深度学习框架，就是库，需要用import来使用(import caffe、import tensorflow)

开源框架优势:

1. 降低了入门的门槛，根据需要选择已有模型，也可以增加层，或者在顶端选择分类器核优化算法
2. 框架也没有完美的，不同框架领域也**不完全一致**
3. 深度学习框架提供了一系列深度学习组件，当需要使用新的算法需要用户自己定义，然后调用深度学习框架函数接口使用用户自定义新算法

### 1.1 核心组件

大部分深度学习框架都包含以下五个核心组件:

1. 张量(Tensor)
2. 基于张量的各种操作(Operation)
3. 计算图(Computation Graph)
4. 自动微分(Automatic Differentiation)
5. BLAS、cuBLAS、cuDNN等扩展包(深度学习框架不依赖于拓展，但在特定功能需要)

### 1.2 张量

> 所有运算核优化算法都是基于张量组成的

标量0阶张量，矢量1阶张量，矩阵2阶张量

可以用一个四维张量表示与1个数据集(N,H,W,C)

优势: 将各种各样数据抽象张量表示，再输入神经网络模型进行后续处理是一种**必要且高效**的策略

处理完成后，还可以方便将张量再转换回需要的格式。

### 1.3 基于张量的操作

对于张量对象，就会有一系列针对这一对象的数学运算和处理过程

其实, **整个神经网络都可以简单视作为了达到某种目的, 针对输入张量进行一系列操作的过程。所谓的学习就是不算纠正神经网络实际输出结果和预期结果之间误差的过程**

处理涉及的范围很广，有简单的矩阵乘法，卷积、池化、LSTM。各个框架支持的张量操作不同

#### 1.4 计算图

前后端之间的中间表示(Intermediate Representations)

> 通过描述输入输出之间的连接关系，最终实现网络结构的可视化效果

一个简单的计算图可以表示为:

![ComputationalGraph](https://i.imgur.com/qjPbqyR.png)

### 1.5 自动微分工具

> 可以将神经网络视为由许多非线性过程组成的一个复杂函数体 f(x)

keras.fit的训练过程就是一个自动微分

### 1.6 拓展包

1. 因为大部分过程基于高级语言实现，特点**运行缓慢**
2. 而低级语言最优化编程难度高，需要利用现成**加速包**

## 二、主流深度学习框架

TensorFlow，Keras，PyTorch，MXNet，Caffe，FastAI，...

主要深度学习框架对比:

![frame_table](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3017661%2F764df060a0403187e17528e14eabad34%2Fdeep-learning-framework-comparison.png?generation=1591387188713236&alt=media)

顶级深度学习的四大框架:

| Name | Front | Commpany |
| ---- | ---- | ---- |
| Tensorflow | Keras | Google |
| PyTorch | FastAI | Facebook |
| MXNet | Gluon | Amazon |
| CNTK | Keras/Gluon | Microsoft |

> MXNet主要用于语音场景，最大的贡献组织百度

本土的深度学习框架:

1. 华为MindSpore，半开源
2. 百度PaddlePaddle，完全开源，**多个领先预训练中文模型**
3. 阿里巴巴XDL(X-Deep Learning)，闭源
4. 小米MACE，闭源

### 2.1 标准化:ONNX

> ONNX深度学习模型的开放格式

是一个深度学习模型的开放格式，可以再不同框架之间转移模型

允许PyTorch模型，使用MXNet运行该模型进行推理

## 三、Tensorflow

TensorFlow是一个采用**数据流图**(data flow graphs)，用于数值计算的开源框架

其中，**节点(Nodes)**在图中表示数学操作，**线(edge)**表示节点间相互联系的多维数据数组(张量)

### 3.1 基本用法

1. 使用图(graph来表示计算任务)
2. 会话(session)的上下文(context)执行图，**计算在session中启动**
3. 使用tensor表示数据
4. 通过变量(Variable)维护状态
5. 使用feed和fetch可以为任意的操作(arbitrary operation)赋值或获取数据

> 图中的节点为op，获取和产生tensor，tensor的维度[B,H,W,C]

### 3.2 构建图(Graph)

#### 1) 创建源op

源op**不需要任何的输入**，主要目的是传输给其他op运算

首先导入包:

```python
import tensorflow as tf
```

接着创建源，这里是创建了三个op，分别是两个常量，以及一个运算

```python
mat_1 = tf.constant([[3.,3.]])
mat_2 = tf.constant([[2.],[2.]])
product = tf.matmul(mat_1, mat_2)
```

#### 2) session启动图

需要创建**会话session**，并在会话中启动图(开启运算)

启动默认图，并运行保存结果导result:

```python
sess = tf.Session()
result = sess.run(product)
```

#### 3) 释放资源

结束会话以后，用session.close()关闭，或者将整体放入with框架内

### 3.2 张量(Tensor)

构建图输出的结果为一个Tensor，包含三个属性(Name、Shape、Type)

1. Name，张量的名称(op添加name属性为op命名)
2. Shape，张量的维度
3. Type，张量的类型，每个张量有**唯一类型**(参考np.dtype)

> 注意保证参与运算的张量类型一致，避免类型不匹配

### 3.3 变量(Variables)

变量Variables维护图执行过程中的状态信息(类似于一般的变量)

例如, 你可以将一个神经网络的权重作为某个变量存储在一个tensor 中. 在训练过程中, 通过重复运行训练图, 更新这个tensor

### 3.4 Fetch/Feed

#### 1) Fetch

Fetch: 取回操作的输出内容

可以使用session对象的run()调用执行图时，传入一些tensor，会取回对应的结果。

比如对于下面的图

```python
num_1 = tf.constant(2.0)
num_2 = tf.constant(3.0)
num_3 = tf.constant(7.0)
inter_p = tf.add(num_2, num_3)
mul_p = tf.multiply(num_1, inter_p)
```

在sessio中传入**更多图**也能获得更多运算中结果:

```python
sess = tf.Session()
result = sess.run([mul_p, inter_p])
```

#### 2) Feed

Feed：使用一个tensor值临时替换一个操作的输出结果

> 给session run过程中图的参数进行传递(**传参**)

对于下面的图

```python
input_1 = tf.placeholder(tf.float32)
input_2 = tf.placeholder(tf.float32)
output = tf.multiply(input_1, input_2)
```

其中input_1和input_2都使用了占位符(placeholder)，**定义了数据模板**

在session run中可以feed入参数

```python
sess = tf.Session()
result = sess.run([output], feed_dict={input_1:[7.0],input_2:[2.]})
```

### 3.5 TensorBot

可视化工具

1. 生成一个具有写权限的日志文件操作对象

    ```python
    writer=tf.summary.FileWriter('logs', tf.get_default_graph())
    writer.close()
    ```

2. 启动tensorboard服务（在命令行启动）

    ```cmd
    tensorboard --logdir logs
    ```

3. 启动tensorboard服务后，复制地址并在本地浏览器中打开

作业

1. keras手写数字识别
2. 从0实现神经网络训练
3. tf实现

## 四、Pytorch

## 五、相关优化算法: BGD SGD

## 六、拓展: Adam
