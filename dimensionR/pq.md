# Product quantization

## 背景

作为降维算法的一种，PQ还是比较出名的，它的经典场景就是图像检索。图像信息不像文字信息有比较具体的字典，可以限定特征的范围，很多维度上都是浮点数，无法像搜索引擎一样建立倒排索引。所以，一些降维也就应运而生。

## KMeans

KMeans是一种很容易想到的算法，我们人为地设定一些中心，然后对于每一个数据，计算到中心的距离，这样原来的高维数据被降到了中心数量的维度。

关于KMeans的算法就不多说了，大家都懂。

## PQ

KMeans的方法足够简单直观，但是也存在一些问题，那就是这样降维会不会太暴力了一些？大家想想觉得也是，于是又想到了一个方法，那就是我把特征拆成一段一段，每一段做一个KMeans，这样效果是不是能好些呢？

这就是PQ算法的核心思想，不过PQ算法对每一段特征采用硬分类的方法，将其分到其中的某一个类中。

下面直接上代码了：
```python
def trainPQ(data): # data 是一个numpy array
  cols = data.shape[1]
  slice = cols / M # 把特征分成M段，每段有slice个特征
  center_set = []
  idx_set = []
  for i in range(M):
    subdata = data[:,i * slice: (i+1)*slice]
    centers = kmeans(subdata, K) # 把这些数据取出来，做kmeans
    idxs = find_closest_idx(subdata, centers) # 对每个数据做硬分类
    center_set.append(centers) # 保存center点
    idx_set += idxs # 保存数据所属的cluster点
  return center_set, idx_set

def query(item, center_set, idx_set):
  feature = []
  # 根据各段的feature找到硬分类的cluster id
  for i in range(M):
    subitem = item(i * slice : (i + 1) * slice)
    idx = find_closest_idx(subitem, center_set[i])
    feature += idx
  find topNbyDistance(feature, idx_set) #正常的距离计算，或者其他加速算法
```

可以看出，整体的思想还是比较简单的，而它的效果也比纯KMeans要好一些。

## PQ的改进

PQ算法也带来了一些问题，因为要把一些feature聚在一起，那么哪些feature聚在一起最好呢？这可就说不清了，于是几个随机化方法诞生了。

1） rand order PQ: 做PQ前，我把feature的顺序随机打乱下，会不会好呢？

2） rand rotate PQ: 我做一个SVD，再做PQ,会不会好呢？

这些方法多多少少有些好处，不过看着还是不那么放心

## OPQ-Optimized版本

TODO

## Reference

http://research.microsoft.com/en-us/um/people/kahe/cvpr13/
