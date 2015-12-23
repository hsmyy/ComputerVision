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

