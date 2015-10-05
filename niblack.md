# Binarization/二值化

### 问题

二值化的问题属于分割问题中的一种形式，分割的主要目的是把一个复杂的图片进行拆解，将待分析的内容与其他信息分离，从而使分析工作更加准确。

二值化则是对分割做了明确的限制，对于一副图片，我们将其分成两类，一类成为前景，也就是这幅图片想表达的内容，另一幅是背景，也就是这幅图片附属的内容。大家很容易理解这个概念，在很多人的到此一游照中，人物和一些核心景色是前景，而其他的花花草草就是背景。说白了，背景就是些可有可无的东西。

为什么要做二值化？对于一些场景，基于先验知识我们已经可以知道图片的内容只有两个类，那么分割的问题就可以归约为二值化，比方说文档图片。

那么二值化和传统分割有什么区别呢？首先二值化的选择范围很窄，对于一个像素不是前景就是背景，逻辑上比多类分割要简单；其次二值化明确了两个类的内容：前景和背景，那么从这个角度也可以添加更多的先验使得效果和效率更好。

我们考虑的图像主要是灰度的文本图像（document image）。

### Threshold/阈值化

对于灰度图的二值化，很容易想到的就是下面的方法：
```python
def threshold(pixel, thresholdVal):
  if pixel > thresholdVal:
    return 1
  else:
    return 0
```

没错，就是这么简单，可以想象，大多数的文档图片都是白纸黑字，那么我们选定某个阈值，把比它黑的像素变成0，其他的变成1，就完成了分离，0代表了前景。

这个方法靠谱么？说实话在理想状态下确实靠谱，因为在没有任何干扰的情况下，背景几乎是白色的，文字几乎是黑色的，以一个阈值进行分离很合理。

但是带来的问题是，阈值该设多少？这个可不好说啊……

### Otsu/大津法

大津同志早在半个世纪前就解决了这个问题，方案就是用算法找寻两个分类的边界。那么怎么找呢？

我们可以想象，假设在一个数据集上有2个类，我们能想到的最直接的方法就是k-means了。我们设k为2，那么代码如下所示：
```python
def meansOf2(img):
  center = [0,255]
  groups = [0] * img.rows * img.cols
  for i in iteration:
    for i,pixel in enumerate(img):
      if abs(center[0] - pixel) < abs(center[1] - pixel):
        groups[i] = 0
      else:
        groups[i] = 1
    for i in range(len(center)):
      subset = [img[i] for i, group in enumerate(groups) if group == i]
      center[i] = sum(subset) / len(subset) # 忽略除0的事情...
```

kmeans做这件事可以么？答案是当然可以，不过这么做复杂度有点略高，因为这还需要若干次的迭代。而大津法则是跳过这个步骤看问题的本质。

由于同类之间的差别比较小，那么它的方差一定不大，异类之间的差别比较大，所以方差一定很小。我们可以把方差这个公式做拆解：
```python
def variation(set): # set = [(val, class)]
  mean = mean([n.val for n in set])
  set0 = [n.val for n in set if n.class == 0]
  set1 = [n.val for n in set if n.class == 1]
  mean0 = mean(set0)
  mean1 = mean(set1)
  return (len(set0) * (mean0 - mean) ^ 2 + len(set1) * (mean1 - mean) ^ 2) / len(set)
```

可以看出，这相当于把一个有N个数的集合变成了2个数的集合，然后求方差，我们的目标是让方差变大，因为如果分得开，那么两个类的均值和整体的均值相差一定会大，故而目标值会变大，反之可以考虑一个极端情况，如果一个类为空，那么这个目标值为0，是最小的。

实际上如果把最后一行的公式展开，可以得到另外一个更为简便的公式：
```python
return len(set0) * len(set1) * (mean0 - mean1) ^ 2 / len(set) / len(set)
```

这样可以减少运算。

下面的问题就简单了，对于一个离散化的图像中，像素值一共有256种选择，我们将256种选择各做一遍，找到那个最大的间隔就可以了，由于中间的结果可以反复利用，实际上这个算法可以非常快。
```python
def otsu(img):
  imgs = bin_sort([pixel for pixel in img])
  
```
