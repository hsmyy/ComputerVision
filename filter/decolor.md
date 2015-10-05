# Decolor/灰度化

### 问题

灰度化也是机器视觉一个很经典的问题。站在数据的角度，就是将一个个三维向量转换成一维向量。根据前人的研究，人们发现人眼对绿色的会比其他颜色更敏感，所以在灰度化的过程中，绿色的权重会占得比较大。经典的公式如下所示：
```python
def decolor(pixel):
  return 0.3 * pixel.r + 0.59 * pixel.g + 0.11 * pixel.b
```

可以看出三种颜色的权值相加为1，在降维的过程中不会造成数字范围的改变。

这个公式存在一个严重的问题，那就是由于不同的颜色在灰度化时会转换成同样的灰度值，那么如果这两种颜色恰好相邻，在转换成灰度之后，这两种颜色会混在一起。

基于这个问题，我们对灰度化有了另外一个约束：尽可能地保留图像的高频细节。

### decolor in opencv

在opencv的photo包中有一个decolor的方法，这个方法来自于香港中文大学的一篇paper，今天的主题也就讲述这篇paper。

首先提出paper的模型：
```python
def decolor2(pixel, weight):
  mVar = [pixel.r, pixel.g, pixel.b, 
          pixel.r * pixel.g, pixel.r * pixel.b, pixel.g * pixel.b, 
          pixel.r ^ 2, pixel.g ^ 2, pixel.b ^ 2]
  return sum([mVar[i] * weight[i] for i in range(mVar)])
```

模型共有9个参数。对于每一次的decolor，我们需要先求出这9个参数，然后再做灰度化。

### 基于对比的优化

那么如何去求这几个参数呢？下面引出他的损失函数：

```python
def loss(img):
  loss = 0
  for pixel1 in img:
    for pixel2 in img:
      loss += (decolor2(pixel1) - decolor2(pixel2) - delta(pixel1, pixel2)) ^ 2
  return loss
```

其中的delta函数如下所示：
```python
def delta(pixel1, pixel2):
  lab1 = convertToLab(pixel1); #转换成Lab坐标系
  lab2 = convertToLab(pixel2); 
  return sign(lab1.l - lab2.l) * sqrt((lab1.l - lab2.l) ^ 2 + (lab1.a - lab2.a) ^ 2 + (lab1.b - lab2.b) ^ 2)
```

delta求的实际上是两个颜色在lab通道上的差距。这个损失函数的形式有点像svm的合页损失，大概的意思是，如果两个颜色有色差，那么它的差距最好和lab的差距相近。lab的l通量代表了光照强度，ab代表了色彩，它经常被用来做色彩相关的分析。从上面的两段代码可以看出这个损失函数是希望通过调整参数使得灰度化后的图像能够保留彩色图像的效果。

文章还从贝叶斯的角度去理解，它把图像中每一对颜色的色差看作以delta(pixel1,pixel2)为均值，某个sigma为方差的高斯分布，所以似然损失函数可以写成：
```python
def likelihood(img):
  loss = 1
  for pixel1 in img:
    for pixel2 in img:
      loss *= exp(-(decolor2(pixel1) - decolor2(pixel2) - delta(pixel1, pixel2)) / 2 / sigma^2)
  return loss
```

### weak color order/弱色阶

这是本文的一大创新之处，也是一个很巧妙的想法。在过去的一些灰度化算法中，色阶往往被定义为强色阶。也就是说，任意两个彩色颜色必然可以分出大小。这样做的好处是可以极大地限制优化的范围（强先验），但是带来的问题是图像的一些细节也会受到损失。而且一个躲不开的问题是，降维后的空间变小，完全保证全色阶也不是一个靠谱的事，所以弱色阶应运而生。

一下是弱色阶函数alpha:
```python
def alpha(pixel1, pixel2):
  if pixel1.r < pixel2.r and pixel1.g < pixel2.g and pixel1.b < pixel2.b:
    return 1.0
  else:
    return 0.5
```

于是新的损失函数变为：
```python
def likelihood(img):
  loss = 1
  for pixel1 in img:
    for pixel2 in img:
      loss *= alpha(pixel1, pixel2) * exp(-(decolor2(pixel1) - decolor2(pixel2) - fabs(delta(pixel1, pixel2))) ^ 2 / 2 / sigma ^ 2) + 
      (1 - alpha(pixel1, pixel2)) * exp(-(decolor2(pixel1) - decolor2(pixel2) + fabs(delta(pixel1, pixel2))) ^ 2 / 2 / sigma ^ 2)
  return loss
```

可以看出，两者的区别，如果alpha函数为1，那么表示两种颜色有明显的排序关系，那么loss函数只有前半部分有效，如果alpha函数为0.5, 那么说明两种颜色的排序关系不明显，那么实际上这两种函数谁先谁后并不是那么重要，所以delta的差值可正可负，从而得出了这个全新的公式。

这个问题的优化相对而言有些复杂，不过大体思路是对weight求偏导，然后利用固定点（fixed point）的方法求解。这部分的内容可以参见paper或者opencv中的代码

### 总结

本文提出了一个问题：常用的灰度化算法无法解决一些相邻色的对比问题，而这个算法可以解决，同时它还解决了过去一些算法在强色阶条件下出现的问题，利用弱色阶解决了问题。

鉴于其代码入选了Opencv的库，所以它的效果还是可以肯定的，作者在matlab上跑一张600*600的图，花了0.8秒，稍微长了些，c的速度尚未测试，如果算法的时间可以控制在10ms以下（视频流实时效果），相信它的应用范围会得到极大的扩展。
