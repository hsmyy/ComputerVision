# Guided Filter

### 问题

这个问题和前面看过的几个文章类似，还是做类似去噪或者去除细节梯度的问题。

文章提取，现在的filter大概分为2类：一类是基于优化的filter，比方说total variation, weighted least square等，这些方法的效果还是不错的，但是算法的时间复杂度普遍比较高，运算时间长。另外一派是guidance map方法，代表方法就是联合双边滤波，双边滤波器有哪些问题呢？首先双边滤波器若是想保证质量，就会比较慢（一些文章测试1M像素的时间在10s），同时还有可能出现gradient reversal problem，也就是梯度方向反转的问题，因为双边滤波会针对像素值金星加权，有可能出现某个像素的相近像素很少的情况，那么像素在过滤后就有可能产生一些变异。在作者的paper中已经展示了这个问题。

另外一个很牛逼的特性是，这个算法的运行时间和filter kernel的size无关。

### 模型

那么guided filter给出的是什么样的模型呢？它所用到的参数和联合双边滤波器相似。

```python
def transform(refImg, alpha, beta):
  ret = array((refImg.rows, refImg.cols))
  for y in refImg.rows:
    for x in refImg.cols:
      ret[y][x] = alpha * refImg[y][x] + beta
  return ret
```

在原始的模型里，alpha和beta表示当前像素所落在的窗口内的alpha，beta参数值。因为整个图像中我们可以找到许多窗口，每一个点实际上属于很多个窗口，因此这里的alpha和beta实际上是多个窗口参数的平均。

其实就是一个线性模型。可以看出，最终的结果和参照图像呈线性关系，因此结果图的梯度和参照图的梯度呈现正相关。

### 优化目标和分析

求解里面参数的方法是采用最小二乘的方法。

```python
def objective(img, ref):
  return (alpha * ref + beta - img) ^ 2 + epsilon * alpha ^ 2
```

作者试图通过intuitive的方法分析目标函数背后的意义，假设我们不考虑正则项，对于下面两种情况：

1） Flat patch:平滑区域，可以设alpha=0, beta=avg(img)满足优化目标，这样最终的输出结果就是原图的平均；

2） High Variance:如果变化很大，那么alpha=1，beta=0，这样就相当于以guided image为最终输出。

从另外一个角度考虑，如果一个window内的patch的方差小于epsilon，那么TA有可能是flat patch,反之就是High variance。

### Filter Kernel 和分析

最终可以得到一个filter kernel的计算公式：
```python
def kernelWeight(img, i, window):
  muK = mean(img(window))
  sigmaK = std(img(window))
  return sum([1 + (img[i] - muK) * (img[j] - muK) / (sigmaK + epsilon) for j in window]) / len(window)
```

从这个公式，我们可以分析出一些东西来：

对于平滑区域，第二项会非常小，那么整体效果就接近low-pass filter

对于边缘区域，如果两个像素一个比mean大，一个比mean小，那么第二项为负，会拉低整个权重，从而使自己的贡献变少。

### 实现

找到的别人实现的一段代码，如下所示：

```c++
cv::Mat GuidedFilterMono::filterSingleChannel(const cv::Mat &p) const
{
    cv::Mat mean_p = boxfilter(p, r); // boxfilter是blur()函数的封装
    cv::Mat mean_Ip = boxfilter(I.mul(p), r);
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p); // this is the covariance of (I, p) in each local patch.

    cv::Mat a = cov_Ip / (var_I + eps); // Eqn. (5) in the paper;
    cv::Mat b = mean_p - a.mul(mean_I); // Eqn. (6) in the paper;

    cv::Mat mean_a = boxfilter(a, r);
    cv::Mat mean_b = boxfilter(b, r);

    return mean_a.mul(I) + mean_b;
}
```
其中输入图片和参照图片是同一张图片。

### 特点

这个算法的一大特点就是快，从上面可以看出它的复杂度是线性的，速度比原始的联合双边滤波器要快。

作者统计：1M gray pixel时间是80ms,1M color pixel时间是300ms。

### 加速算法

作者在多年后提出了一个加速算法，主要是对图片先做了down-sampling,然后做up-sampling的滤波，速度又能有很大的提高。
具体的解法可以参见paper内容。

### Reference

[PPT](http://research-srv.microsoft.com/en-us/um/people/kahe/eccv10/eccv10ppt.pdf)
[Paper](http://131.107.65.14/en-us/um/people/jiansun/papers/GuidedFilter_ECCV10.pdf)
