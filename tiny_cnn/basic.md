# 基本概念

Tiny Cnn作为一个小而美的lib，在CNN的浪潮之中还是有TA的存在价值的。相比其他框架，TA足够小，小到可以在了解基本原理后快速上手了解和应用实现

我使用的版本不是来自官方版本，而是一个民间版本，这个版本可以很好地在Mac上进行编译使用，唯一要做的就是装一个boost。

### 整体架构

这个lib大概可以分为几个部分：

1）边角代码

image.h:将图片保存成uchar vector封装起来

util.h:一些功能函数，用到再看

deform.h:本质就是用伯努利概率加胡椒噪声

product.h:SSE和VAX加速用的

mnist_parser.h:读取mnist数据

config.h:?

2) 基础代码

activation_function.h：这里面介绍了激活函数：sigmoid(据说CNN界不怎么用TA)，rectified_linear unit（新贵）,tan_h。

loss_function.h：介绍了常用的loss函数：mse和cross_entropy

optimizer.h: 集中经典的优化方法（有空补完）

3）网络层代码

layer.h：基础层

fully_connected_layer.h，fully_connected_dropout_layer.h：全连接层

partial_connected_layer.h：部分连接层

convolutional_layer.h：卷积层

average_pooling_layer.h,max_pooling_layer.h：两个pooling层

dropout.h：dropout特性

3）网络层

network.h：把网络串起来
