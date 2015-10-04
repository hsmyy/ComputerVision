# Decolor/灰度化

### 问题

灰度化也是机器视觉一个很经典的问题。站在数据的角度，就是将一个个三维向量转换成一维向量根据前人的研究，人们发现人眼对绿色的会比其他颜色更敏感，所以在灰度化的过程中，绿色的权重会占得比较大。经典的公式如下所示：
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
  mVar = [pixel.r, pixel.g, pixel.b, pixel.r * pixel.g, pixel.r * pixel.b, pixel.g * pixel.b, pixel.r * pixel.r, pixel.g * pixel.g, pixel.b * pixel.b]
  return sum([mVar[i] * weight[i] for i in range(mVar)])
```

模型共有9个参数。对于每一次的decolor，我们需要先求出这9个参数，然后再做灰度化。

###
