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

#
