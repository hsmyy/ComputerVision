# Rolling Guidance Filter

### 问题

这次的问题依然是去噪，可以说这一次的去噪问题实际上和本章的另一个问题Total Variation是类似的。不过这篇paper主要的应用场景是去掉texture，关于texture的问后面再说。

### Bilateral Filter/双边滤波

相信每个人都已经对高斯滤波很熟悉了，简单地写个伪代码：
```python
def gaussian(img, y, x, window, sigma):
  sum = 0
  pixel = 0
  for dy in range(-window, window):
    for dx in range(-window, window):
      weight = exp(-(dy ^ 2 + dx ^ 2) / (2 * sigma ^ 2));
      sum += weight
      pixel += weight * img[y + dy][x + dx]
  return pixel / sum
```

可以看出，高斯的滤波主要根据空间进行加权，越近的权重越大。而双边滤波则是同时进行两个方面的加权，那么另一个是像素值。
```python
def bilateral(img, y, x, window, sigmaS, sigmaI):
  sum = 0
  pixel = 0
  for dy in range(-window, window):
    for dx in range(-window, window):
      weight = exp(-(dy ^ 2 + dx ^ 2) / (2 * sigmaS ^ 2) - (img[y + dy][x + dx] - img[y][x]) ^ 2 / (2 * sigmaI ^ 2) )
      sum += weight
      pixel += weight * img[y + dy][x + dx]
  return pixel / sum
```

从weight值就可以看出其中的不同，现在只有越近且像素值相近的像素才会有高的贡献。

为什么要这样设计？我们可以想象一个场景，对于某个边缘像素，一边与它完全不同，一边与它十分相近。对于高斯滤波，它会把所有相近的像素赋予高权重，而双边滤波只会给和自己相近的像素赋予高权重。

那么我们可以得到一个结论，双边滤波对于边界的保留效果比高斯滤波要好。

### Joint Bilateral Filter

聪明的人类在双边滤波器的基础上又增加了新的变化，我们可以看到在之前的双边滤波中，权重的产生和像素值都来自于同一张图，那么如果来自不同的图呢？

这就是联合双边滤波的思想，我们可以用一张图做权重，一张图做滤波像素值，于是代码变成了：
```python
def jointBilateral(img, refImg, y, x, window, sigmaS, sigmaI):
  sum = 0
  pixel = 0
  for dy in range(-window, window):
    for dx in range(-window, window):
      weight = exp(-(dy ^ 2 + dx ^ 2) / (2 * sigmaS ^ 2) - (refImg[y + dy][x + dx] - refImg[y][x]) ^ 2 / (2 * sigmaI ^ 2))
      sum += weight
      pixel += weight * img[y + dy][x + dx]
  return pixel / sum
```

前面这些都是铺垫，下面来到了我们要介绍的滤波器。

### 基本思想

下面回到前面paper里的概念，paper里面提到了edge scale的概念，也就是说图像中的每一个边缘都有不同的尺度，有的边缘小且短，有的边缘大且粗，它们可以看作不同的尺度。

于是算法的理念是，一些小而短的边缘属于texture，并不是主要的高频信息，而另外一些大而粗的边界往往是物体的主要边界，这些边界对于分割物体，主体识别等任务是十分重要的，所以为了排除一些细节的干扰把texture去掉，留下核心的边缘信息会对未来的工作有很大的帮助。

换个思路，在一些场景中，噪声就可以看作是小而短的texture，其他的信息在尺度上会大于那些噪声，也就是主体边缘了，所以去也算是这个问题的一个类型。

这个思路是不是和TV很像呢？

### 高斯的尺度参数

前面提到了高斯滤波，但是没有详细地讲述其中的一个参数，sigma，这个参数越大，平滑的效果越厉害，反之则越不起效果。我们假设当sigma设置为某个值时，所有的texture全部被消除，或者基本消除，那么我们就把这个参数叫做scale parameter，这是区分两种边缘的临界参数。

下面的过程中，我们将用到这一步。

### 整体算法

下面是整体算法：
```python
def rollingGuidanceFilter(img, sigmaS, sigmaI, window, iteration):
  refImg = [[gaussian(img, y, x, window, sigmaS) for x in img.cols] for y in img.rows]
  for i in range(iteration):
    refImgNew = [[jointBilateral(refImg, img, y, x, window, sigmaS, sigmaI) for x in img.cols] for y in img.rows]
    refImg = refImgNew
  return refImg
```

其实这个算法看上去并不复杂，但是其中蕴含的道理却十分精妙。

### 核心思想

前面我们提到了高斯的scale parameter,在这个参数下所有的小边缘将被消除。那么我们拿着这张图作为联合双边滤波的参数图会发生什么？

对于小边缘，由于在refImg中已经消除，即使经过原图的加权，由于高频细节已经消除，得到的依然是平滑后的像素

对于大边缘，由于在refImg中没有完全消除，经过原图的加权，一些和原图中相似的像素点由于没有被高斯滤波消除，会重新叠加到当前像素上，因此当前像素即使被高斯平滑，也会在不断的迭代中恢复回来。

一般的实验中，如果尺度设计合理，5到6轮之后就会得到好的结果。

### 扩展

这个算法的另外一个贡献是它的框架，再做完第一步的高斯滤波后，这张图可以做guidance map，那么一些相近的算法也可以套上来，比方说guided filter,domain transform filter等。

guided filter的使用相对比较直观：
```python
def rollingGuidanceWithGuidedFilter(img, sigmaS, window, epsilon, iteration):
  ref = [[gaussian(img, y, x, window, sigmaS) for x in img.cols] for y in img.rows]
  for i in range(iteration):
    newRef = GuidedFilter(img, ref, window, epsilon)
  return newRef
```
关于guided filter的内容可以参考本系列介绍guided filter的文章。

domain transform filter的使用稍微有点晦涩，这里可以稍微拆开一点，具体内容还是见本系列介绍dt filter的文章。
```python
def rollingGuidanceWithDomainTransformFilter(img, sigmaS, window, iteration):
  ref = [[gaussian(img, y, x, window, sigmaS) for x in img.cols] for y in img.rows]
  for i in range(iteration):
    transformedDomainX, transformedDomainY = dt(ref)
    newRef = img
    for j in range(3): # 论文上说3轮就够了
      newRef = horizontalBilateral(newRef, transformedDomainX)
      newRef = verticalBilateral(newRef, transformedDomainY)
  return newRef
```

最快的就是Domian Transform Filter,它的加速可以达到1M pixel 50ms，算是非常快了。

### 总结

Rolling Guidance Filter的核心就是利用联合双边滤波解决问题，主要的思想就是将边缘信息进行分类，首先保证去除texture信息，然后逐步恢复主体边缘。

### 番外

关于这个算法，它可以把它的高斯滤波部分进行加速，加速的方法是采用一种特殊的数据结构，想搞清这个问题的童鞋可以去看一篇长论文，里面的一章讲得很清楚。

### Reference

[Rolling Guidance Filter](http://www.cse.cuhk.edu.hk/leojia/projects/rollguidance/)
