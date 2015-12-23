# 优化入门

## 算法概览

代码入口：https://github.com/tamland/non-linear-optimization

## 代码接口

```python
def f(x): #函数本身
def df_dx1(x): #∂f/∂x1偏导
def df_dx2(x): #∂f/∂x2偏导
def df(x): return [df_dx1(x), df_dx2(x)] #x的整体梯度
def df_dx1_dx1: #∂f/∂x1 * ∂f/∂x1二阶偏导
def df_dx1_dx2: #∂f/∂x1 * ∂f/∂x2二阶偏导
def df_dx2_dx2: #∂f/∂x2 * ∂f/∂x2二阶偏导
def df2(x): return [ [df_dx1_dx1(x), df_dx1_dx2(x)], [df_dx1_dx2(x), df_dx2_dx2(x)] ]

steepest_descent(f, df, x, max_iterations, precision) #经典的梯度下降法
newton(f, df, df2, x, max_iterations, precision) #牛顿法
quasi_newton(f, df, x, max_iterations, precision) #拟牛顿法
conjugate_gradient(f, df, x, max_iterations, precision) #共额梯度法
```

## 算法表现

```python
def f(x): 
  return 100 * (x[1] - x[0]^2)^2 + (1-x[0])^2
```

* steepest descent迭代满100轮未收敛
* newton迭代14轮收敛
* 拟牛顿迭代21轮收敛
* 共额梯度迭代11轮收敛

## 算法的两个部分

 * 计算梯度
 * 计算步长

## 步长计算理论

### Wolfe condition && Wolfe strong condition

Wolfe condition要求步长能够满足足够的下降。

Wolfe strong condition, 也叫curvature condition, 它通过计算梯度来判断下一步的计算（大梯度可以接着算，小梯度可以歇歇）

```python
# step表示步长
# grad表示梯度
g = lambda step: f(x + step * grad)
gd = lambda step: f(x + step * grad) * grad

# c1通常设为1e-4
def wolfe_normal(x): # 公式定义
  return f(x + alpha * step) <= f(x) + c1 * alpha * df(x) * step

def wolfe(g, gd, step, c1): # 代码中的wolfe
  return g(step) <= g(0) + c1 * alpha * gd(step)
  
# c2通常设为0.9
def wolfe_strong_normal(x): # 公示定义
  return abs(df(x + alpha * step)) >= c2 * df(x) * step

def wolfe_strong(g, gd, step, c2):
  return abs(gd(alpha)) <= -c2 * gd(0)
```

### simple_backtracking/简单回退

```python
def simple_backtracking(g, gd, step, c1, c2):
  rate = 0.5
  while not(wolfe(g, gd, step, c1) or wolfe_strong(g, gd, step, c2)):
    step = rate * step
  return step
```

### interpolation

```python
```

## steepest descent/最速下降法

```python
def steepest_descent(f, fd, x, max_iterations, precision, callback):
  for i in range(0, max_iterations): # 进行有限的循环迭代，也可以把下面的if判断移过来
    direction = - fd(x)  # 求方向，也就是当前的负梯度（与切线方向正交，代表了最快的下降方向）
    alpha = find_step_length(f, fd, x, 1.0, direction, c2=0.9)  # 求步长
    x = x + alpha*direction # 更新参数
    callback(i, direction, alpha, x)
    if linalg.norm(direction) < precision: # 如果解足够精确，则停止
      break
  return x
```

## newton/牛顿法
```python
def newton(f, fd, fdd, x, max_iterations, precision, callback):
  for i in range(1, max_iterations):  # 迭代循环
    gradient = fd(x)  # 求1阶导向量
    hessian = fdd(x)  # 求2阶导矩阵

    direction = -linalg.solve(hessian, gradient) # 因为方向等于gradient/hessian,也就相当于一个Ax=b的解方程问题，A是hessian
    alpha = find_step_length(f, fd, x, 1.0, direction, c2=0.9) # 求步长
    x_prev = x # 保存上一步的结果
    x = x + alpha*direction # 更新参数

    callback(i, direction, alpha, x)

    if linalg.norm(x - x_prev) < precision: # 如果没有足够大的更新，则结束
      break
  return x
```

## quasi-newton/拟牛顿
```python
def quasi_newton(f, fd, x, max_iterations, precision, callback):
  I = identity(x.size)
  H = I
  x_prev = x
  f_prev = f
  fd_prev = fd

  for i in range(1, max_iterations):
    gradient = fd(x)
    direction = -H * matrix(gradient).T
    direction = squeeze(asarray(direction))

    alpha = find_step_length(f, fd, x, 1.0, direction, c2=0.9)
    x_prev = x
    x = x + alpha*direction

    callback(i, direction, alpha, x)

    if linalg.norm(x - x_prev) < precision:
      break

    s = matrix(x - x_prev).T
    y = matrix(fd(x) - fd(x_prev)).T
    rho = float(1 / (y.T*s))
    H = (I - rho*s*y.T)*H*(I - rho*y*s.T) + rho*s*s.T
  return x
```

## conjugate gradient/共额梯度
```python
def conjugate_gradient(f, fd, x, max_iterations, precision, callback):
  direction = -fd(x)
  gradient = None
  gradient_next = matrix(fd(x)).T
  x_prev = None

  for i in range(1, max_iterations):
    alpha = find_step_length(f, fd, x, 1.0, direction, c2=0.1)
    x_prev = x
    x = x + alpha*direction

    callback(i, direction, alpha, x)

    gradient = gradient_next
    gradient_next = matrix(fd(x)).T

    if linalg.norm(x - x_prev) < precision:
      break

    BFR = (gradient_next.T * gradient_next) / (gradient.T * gradient)
    BFR = squeeze(asarray(BFR))

    direction = -squeeze(asarray(gradient_next)) + BFR*direction
  return x
```
