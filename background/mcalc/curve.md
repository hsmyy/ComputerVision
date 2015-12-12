# 参数曲线

### 参数线定义

把一条线想象成一个点的运动轨迹，那么就存在另外一个分量t，用于构建时间和位置的关系：
```python
def getPoint(t):
  return [x(t), y(t)]
```

### 一些参数曲线例子

抛物线：
```python
def parabolic(t, vx, vy): # vx,vy是抛出物体时的x,y方向速度
  return [vx * t, -0.5 * g * t * t + vy * t]
```

圆：
```python
def circle(t, r): # r是圆的半径
  return [r * cos(t), r * sin(t)]
```

椭圆：
```python
def ellipse(t, a, b): # a,b是椭圆两个轴的长度
  return [a * cos(t), b * sin(t)]
```

cycloid/摆线：

一个半径为r的圆向前滚，圆上某一点划过的轨迹
```python
def cycloid(y, a, theta):
  return [a * theta - a * sin(theta), a - a * cos(theta)]
```

### velocity and acceleration/速度与加速度：

```∆r＝(∆x, ∆y)```表示位置的偏移，```∆t```表示时间的偏移,于是```∆r/∆t```表示速度。

当```∆t```趋近于0时，∆r的方向和曲线的方向相切。在几何上称为当前点的切线向量。此时的速度表示为```dr/dt```。

加速度表示为速度在∆t的微分。

### arclength/弧长

对于曲线上的弧线s, 有

```python
ds/dt=norm(V)
```

也就是速度的大小。所以

```python
ds/dt=sqrt(dx * dx + dy * dy)
```

对于单位切线向量
```python
T=V/norm(V)
```
有
```python
V=ds/dt * T 
T = V / (ds/dt)
```

从几何上考虑，∆s不小于∆r，只有当∆t趋近于0时，弧线的微分和位置偏移的微分相等，也就是：
```python
ds/dt = norm(dr/dt)
```



