# Guided Filter

### 问题

这个问题和前面看过的几个文章类似，还是做类似去噪或者去除细节梯度的问题。

所不同的是，這一次所面对的目标有些变化。本文的作者主要是基于双边滤波器进行优化。双边滤波器有哪些问题呢？首先双边滤波器若是想保证质量，就会比较慢（一些文章测试1M像素的时间在10s），同时还有可能出现gradient reversal problem，也就是梯度方向反转的问题，因为双边滤波会针对像素值金星加权，有可能出现某个像素的相近像素很少的情况，那么像素在过滤后就有可能产生一些变异。在作者的paper中已经展示了这个问题。

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

其实就是一个线性模型。可以看出，最终的结果和参照图像有很强的相关性，而且结果图的梯度和参照图的梯度呈现正相关。

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

### 加速算法

作者在多年后提出了一个加速算法，主要是对图片先做了down-sampling,然后做up-sampling的滤波，速度又能有很大的提高。
具体的解法可以参见paper内容。
