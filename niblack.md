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
  pixels = [0] * 256
  for pixel in img:
    pixels[pixel] += 1
  sum = sum([pixel for pixel in img])
  bestThreshold = 0
  bestVal = 0
  sum0 = 0
  count0 = 0
  len = img.rows * img.cols
  for i in range(256):
    sum0 += pixels[i] * i
    count0 += pixels[i]
    sum1 = sum - sum0
    count1 = len - count0
    loss = count0 * count1 * (sum0 / count0 - sum1 / count1) ^ 2 / len / len # 忽略除0
    if loss > bestVal:
      bestVal = loss
      bestThreshold = i
  # 最后完成二值化
  return [1 if pixel > bestThreshold else 0 for pixel in img]
```

可以看出，这个算法在效果上和KMeans差不多，但效率大概和1轮迭代的kmeans差不多。

那么这个算法的效果怎么样？在满足上面的所有假设的情况下，效果很好。当然现实往往不是那么完美……

### 现实中的二值化问题

现实中的二值化问题究竟是什么样的呢？在现实中我们常常需要面对下面的问题：

1. 光照不均匀。光线照在一张图上有明有暗，会导致有些字的周围是白色很清晰，而有些字的周围被阴影笼罩，很显然，otsu算法不胜任这个问题。
2. 拍摄有问题。其结果和上面的类似。

总之，直接使用otsu算法是不太靠谱的。

### Niblack

上面的算法可以叫做全局的二值化，也就是说一副图像上的所有像素都按照同样的方法做二值化，这样对于上面提到的问题显然是不太合理的，于是前人很自然地想到：为什么不把全局的算法改成局部的算法呢？

于是乎一批局部算法应运而生，其中的经典当属Niblack，可以说这个算法的效果依然存在局限性，但是它的效果之好让现在的研究人员依然引用它坐二值化算法中的某一步。

我们先考虑一个问题，局部的otsu算法靠不靠谱？
```python
def localOtsu(img, window):
  res = [[0] * img.cols ] * img.rows
  for y in range(img.rows):
    for x in range(img.cols):
      localImg = img(y - window / 2, x - window / 2, window, window) #这句是说我们取一个以y,x为中心，长宽为window的子矩阵。
      localRes = otsu(localImg)
      res[y][x] = localRes[window / 2][window / 2]
  return res
```

我想在一些场景下它是靠谱的，当选中的局部窗口中只包含两种类型的像素,otsu的效果还算可以，但是一旦数目增多，前景背景和阴影全部出现，我想otsu算法不见得可以把阴影和背景分到一起去。另外一个问题是，otsu算法貌似在这个场合下速度会变慢不少。

niblack的方法呢？
```python
def niblack(img, window, k):
  res = [[0] * img.cols ] * img.rows
  for y in range(img.rows):
    for x in range(img.cols):
      localImg = img(y - window / 2, x - window / 2. window, window)
      threshold = mean(localImg) + k * std(localImg)
      res[y][x] = 1 if img[y][x] > threshold else 0
  return res
```

可以看出，和我们上面提出的算法只有一点差距。而且，这个算法实际上有提速的算法。

那么这个算法究竟好在哪里？

答案是好不了多少……

但是，它的速度比前面的局部otsu方法要快。

niblack算法的特点决定了它的功能。我们可以想象上面的threshold计算公式。前半部分表示一个基准位置，也就是说这个threshold不会小于平均值，而k的大小（一般取正数）决定threshold会向大的数字偏离多少，如果对于一张黑底白字的图片，那么k越大背景的精确度越大，前景的召回越大。所以niblack在实际中一般用来做精确定位背景和大概圈定前景的工作。

### 终极问题：二值化的完美方案？

二值化没有完美方案，更复杂靠谱的方案下回再说。
