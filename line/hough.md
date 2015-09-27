# Hough Transform/霍夫变换

霍夫变换是直线检测中的明星算法，我认为有两个原因，第一是它的想法十分独到，一个空间的转换就使得问题得到了很好地解决，第二就是大名鼎鼎的opencv实现了它，速度效果都还算不错。

### 直线检测

为什么要做直线检测？在有些场景下，我们需要找到一张图像上的直线，比方说对于一张名片的照片，我们假设整张名片都被拍摄到了照片里面，那么一个问题就是：名片的范围在哪里？

我们再加一个假设，那就是我们拍摄的照片是大众普遍了解的长方形名片，奇葩形状暂不考虑。对于人眼来说，识别轮廓十分简单，对于计算机来说，识别这个边界就需要一些方法，而其中的一种解决方法就是用霍夫变换。

下面我们把问题做一个定义：

*直线检测：在一个二值图中，每一个像素点的亮度值只可能等于1（前景）或者0（背景），找出由1组成的最长的N条直线*

### 坐标系变换

上面这个问题其实在一些公司的面试中已经出现过，我相信一些人会想到利用极坐标的方式解决问题：
···python
def lineCheck(mat):
  dots = findAllForegroundDotInMat(mat) # 找到所有前景点
  lineMap = map() # key为直线方程，val为贯穿的点数
  for i in range(len(dots)):
    for j in range(i + 1, len(dots)):
      k, b = fitLine(dots[i], dots[j]) # 利用dots[i], dots[j]两点求一条直线
      if map.exist((k,b)):
        continue
      count = 0
      for k in range(len(dots)):
        if k != i and k != j:
          isInLine = isThroughLine(k, b) # 判断点是否穿过前面求出的直线
          if isInLine:
            count += 1
      if count > 0:
        map.insert((k,b),count)
  # 当求出所有的线和贯穿的点数后，剩下的任务就是找出最大的N个返回，这里不再赘述。
```

上面的代码只为表达思路，如有错误请指出并谅解。

可以看出，上面的问题复杂度为O（n^3)，复杂度还蛮高的。
