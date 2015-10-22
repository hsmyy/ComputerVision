# Differential Geometry/微分几何入门

### Intro/序

本文来自于一本书的翻译，这本书是做图形学Mesh人看的书，但是其中介绍的基本知识倒是和图像互通的。

### Curves/曲线

对于一些基本概念，我们需要从简单到复杂慢慢推导，首先是曲线。这里考虑的是在2维平面下的1维可微manifold流型。我们可以定义一个定义域为[a, b],值域为R2的函数:
```python
def curve(u):
  if u in [a,b]:
    return [x(u), y(u)]
  else:
    return None
```
其中x(u),y(u)是两个函数，比方说对于一个半圆的曲线，可以进行如下的定义：
```python
def x(u):
  return u
def y(u):
  return sqrt((b - a) ^ 2 - (u - a) ^ 2)
```
可以看出上面的函数是可微的。

### Tangent Vector/切线向量

tangent vector是指曲线上某一点的一阶导数：
```python
def tangentVector(u):
  if u in [a,b]:
    return [dx(u), dy(u)]
  else:
    return None
```
同理，我们可以求出dx()和dy()。假设我们把曲线想象成动力学中的物体，那么曲线中的点相当于物体运动的轨迹，那么该点的tagent vector相当于该点的速度方向。

同样可以定义一个点的normal vector/法线向量：
```python
def normalVector(u):
  tv = tangentVector(u)
  tv = rotate90degree(tv) // 转动切线向量90度
  tvNorm = norm(tv) // 求向量的范数
  if tvNorm != 0:
    return tv / tvNorm
  else:
    return None
```

### Arc Length/弧长

前面提到的对于曲线的定义有一个问题，就是对同样的一根曲线线段，我们可能拥有不同的定义方法，例如：
```python
def curve1(u):
  if u in [0, 1]:
    return [u, u]
  else:
    return None

def curve2(u):
  if u in [0, 1]:
    return [u ^ 2, u ^ 2]
  else:
    return None
```
这两条曲线实际上都是从(0,0)到(1,1)的一条直线，但是由于采用不同的定义方法，他们的内在性质是不同的，下面就来讨论第一个内在性质——弧长。

弧长可以用曲线的切线向量在作用域的积分表示：
```python
def arcLength(u): //从a到u的弧长
  return integral[ from a to u: tangentVector(t) dt]
```

因为这个性质，tangent vector实际上可以做曲线上的metric/测度，也就是说任意两点的弧长等于这两点在曲线上的距离，而这个距离是通过计算切线向量的积分得到的。

同时，我们可以把原来的2维表示转化1维，同时不会改变曲线任意两点的距离，那种感觉好像是把一条曲线拉直了一样:
```python
def curveArcVersion(u):
  return arcLength(u)
```

通过这个函数，我们可以把2维平面的参数转换到1维表示，经过这个转换，我们可以看出来上面的例子里，curve1和curve2的弧长公式会变得不一样，这样两个公式将产生差别。

### 曲率

如果采用弧长的方式表示曲线，那么我们可以用其表示曲率：
```python
def curvature(u):
  return ddx(arcLength(u))
```

曲率可以用来表示曲线与当前切线向量的偏离度。曲率也是有方向的，朝一个方向是正曲率，那么另一个方向就是负曲率。

这里还有一个概念叫osculating circle/密切圆，它的概念是用一个圆来表示当前点的曲率，而这个圆是和当前点附近点贴合最好的圆。关于密切圆的概念可以自行查找。
