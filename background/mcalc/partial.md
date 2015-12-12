# 偏微分

### Level Curves, Contour Plots/ 等高线

在3D图是除了可以画一个三维模型之外，还可以化成等高线的样子。

### 偏微分

函数在某个方向的偏微分相当于保持其他方向不变，求函数值和这个方向的变化关系。

### 切平面

三维函数z=f(x,y)上一点(x0,y0,z0)的切平面是过这一点且包含∂z/∂x和∂z/∂y两个向量的平面。
```python
def tangentPlane(x,y): #已知x,y求z
  return (∂z/∂x)(x0) * (x - x0) + (∂z/∂y)(y0) * (y - y0) + z0
```

### 线性近似

主要是通过泰勒一阶展开得到

### Critical Points/临界点：

临界点是一阶偏导为0的点：∂z/∂x=0且∂z/∂y=0，极大极小值是临界点。

对于单变量函数，当一阶导数为0，就需要计算二阶导数，如果二阶导数大于0，那么说明在局部是严格递增的，说明该点前一刻的导数为负，后一刻为正，说明前后都比当前点大，反之二阶导数小于0，就是极大点。

对于多变量函数，则有A=fxx', B=fxy', C=fyy',有：
```python
def judgePoint(A, B, C):
  if A * C - B * B < 0:
    return saddlePoint
  if A > 0 or C > 0:
    return minimumPoint
  else:
    return maximumPoint
```

### 全微分
```python
dz/dt=(∂z/∂x)dx/dt + (∂z/∂y)dy/dt
```

### gradient/梯度：

梯度就是由偏微分组成的向量▽z=<∂z/∂x, ∂z/∂y>,它和level curve的正交。

### 方向偏导

某个点在某个方向上的偏导相当于该点梯度和这个方向的单位向量的点积

dz/dx(U) = ▽z * U

### Lagrange Multipliers/拉格朗日乘子

拉格朗日问题：最小（最大化）w=f(x,y,z) ,限定g(x,y,z)=c

拉格朗日答案：极值点满足：▽f=lambda * ▽g 且g(x,y,z)=c

几何上的解释：极值点会同时落在f和g的等高线上，所以两个等高线关于这一点相切，而各自的梯度与等高线垂直，所以两个梯度是平行的。

### 受限问题的求导

对于w=f(x,y), 限定g(x,y)=c,求dw/dx

方法1：
```python
def method1():
  # 对f,g分别求对x的导，由于讲y看作应变量，所以对y求导会得到dy/dx
  # 两个求导公式融合，抵消掉dy/dx,得到最终结果
```
方法2：
```python
def method2():
  # 对f,g分别求全微分
  # 将两个公式的dy抵消掉
```
