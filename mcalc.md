# 多变量微积分

课程来源：http://open.163.com/movie/2010/8/P/F/M6TUC9K75_M6TUHCEPF.html

### 第一课：向量基础

向量定义：1）方向，2）长度

向量的加法和点乘：
```python
def vecAdd(a, b):
  if len(a) != len(b):
    return []
  return [a[i] + b[i] for i in len(a)]

def vecMScalar(a, b):
  return [a[i] * b for i in len(a)]
```

向量的长度（norm）:
```python
def vecNorm(a):
  return math.sqrt(sum([a[i] * a[i] for i in len(a)]))
```

向量的点积：
```python
def vecDotProduct(a, b):
  # omit length check
  return sum([a[i] * b[i] for i in len(a)])

def vecAngle(a, b):
  return acos(vecDotProduct(a,b) / vecNorm(a) / vecNorm(b))
```
对于第二种方法，可以通过余弦定理推导：三角形三边a,b,c，c^2=a^2+b^2-2abcos(angle)

向量的点积还可用于求一个向量a在另一个向量方向b的分量，它等于vecDotProduct(a,b / vecNorm(b))

### 第二课

已知平面上两个向量，求两个向量构成的平行四边形的面积？等于两个向量组成的行列式的绝对值。(推导很精妙)
```python
def det2(a, b):
  return a[0] * b[1] - a[1] * b[0]
```
在三维空间的三个任意向量的行列式等于它们组成的平行六面体的体积：
```python
def det3(a, b, c):
  return a[0] * det2(b[1:], c[1:]) - a[1] * det2(b[0,2],c[0,2]) + a[2] * det2(b[:2], c[:2])
```

三维平面上的两个向量可以做叉积，结果仍是一个向量：
```python
def vecCrossProduct(a,b):
  return [det2(a[1:], b[1:]), det2(a[0,2],b[0,2]), det2(a[:2], b[:2])]
```
叉积的norm等于向量的面积，方向垂直于向量.

叉积的右手定则：食指指向a,然后弯向b，竖起大拇指的方向是叉积的方向。

三维空间三个向量组成的平行六面体的体积还可以等于任意两个向量的叉积和第三个向量的点积（推导很精妙），它和三个向量的行列式相等。

### 第三课

已知两个不共线的向量构成一个平面，求另一个向量是否与之共平面？可以转化为三个向量组成的平行六面体的体积为0

其他关于矩阵的东西略过了。

### 第四课

略

### 第五课

```python
class Plane:
  public _init_(self, vec, offset):
    this.vec = vec
    this.offset = offset

class Line:
  public _init_(self, direction, offset):
    this.direction = direction
    this.offset = offset
```
直线方程可以用两个点表示，也可以用约定某个时间t来表示：

x(t) = at + b

y(t) = ct + d

z(t) = et + f

判断一个点在平面的上面还是下面：
```python
# return: 1 means up, 0 means on plan, -1 means down
def judgeDotPlaneRelationship(dot, plane):
  dotProduct = vecDotProduct(dot, plane.vec)
  if dotProduct > offset:
    return 1
  elif dotProduct == offset:
    return 0
  else:
    return -1
```

摆线的计算公式（推导得很精彩）
```python
def cycloid(theta):
  return [a * theta - a * sin(theta), a - a * cos(theta)]
```

### 第六课

速度：等于位移/时间

加速度：等于速度/时间

弧长：等于速度在时间上的积分

从开普勒第二定律推导到行星的加速度与行星到太阳的方向同向（精彩）

### 第八课

多变量函数的画图问题，采用等高图所带来的好处，偏微分，偏微分与等高线的关系

