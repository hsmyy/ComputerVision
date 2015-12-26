# Steepest Descent/最速下降法

## 背景

最速下降法是一个基础的优化算法，它简单但被广泛利用，搞懂它很有用。

## 原理

在微积分中，我们知道一个点的梯度向量代表了当前向量的行动趋势，如果大于0，表示往前走会越走越大，如果小于0，表示往前走会越走越小，如果等于0，表示这是一个临界点。

于是在优化问题中（求最小值），我们就可以利用负梯度做方向向量，因为如果梯度大于0，说明往前走会变大，那就往回走，负梯度的值正好是负的；如果梯度小于0，那就往前走，负梯度的值正好是正的。

优化问题里面有两个主要的问题，一个是方向，一个是步长，解决了方向，我们来看看步长。首先定义一些变量：
```python
def e(iteration):
  return x[iteration] - x # 后面的x表示ground truth
def r(iteration): #残差
  return b - A * x[iteration] # 残差实际上就是当前点负梯度
```

### 定理1：在最速下降法中，每一步的残差r(n)（负梯度）都与下一步的残差r(n+1)正交。

为什么呢？因为如果r(n+1)与r(n)没有正交，那么r(n+1)一定存在一个朝r(n)方向的分量，说明在上一轮r(n)并没有下降到方向所在的最优点，所以得出结论错误。 （证毕）

既然这样，我可以根据下面的一组公式得到
```python
r(1) * r(0) = 0 #这里是点乘，为了方便就这么写了
=> (b - A * x[1]) * r(0) = 0  #根据残差公式展开
=> (b - A * (x[0] + step * r(0))) #根据更新公式定义

=> step = (r(0) * r(0)) / (r(0) * A * r(0))
```

这样就得到了步长的公式。我们就得到了整体的算法
```python
def steepestDescent(A, b, x0, max_iteration): # 求解Ax=b,其中x代表一个初始解
  r = []
  x = [x0]
  step = []
  for iter in range(max_iteration):
    r[iter] = b - A * x[iter]
    step[iter] = (r[iter] * r[iter]) / (r[iter] * A * r[iter])
    x[iter + 1] = x[iter] + step[iter] * r[iter]
  return x[-1]
```

这个算法显然不是最快的，至于一些快速的计算方法不再此考虑了。

## 收敛性

从直觉上想，每次都选择最陡峭的方向，收敛应该比较快吧，其实不然。下面给出两个收敛极快的特殊情况。

### 定理2：如果e(n)是A的特征向量，那么Steepest Descent可以做到一步内收敛。

```python
# 如果e(n)是A的特征向量，那么有
A * e(n) = lambda * e(n)      #(1)

# 根据上面的公式有
e(n) = x(n) - x
=> -A * e(n) = A * x - A * x(n) = b - A * x(n) = r(n)
#所以
r(n) = -A * e(n) = -lambda * e(n)           #(2)

#把公式1的左右乘以lambda，则有
A * lambda * e(n) = lambda * lambda * e(n)
=> A * r(n) = lambda * r(n)      #(3)

#根据公式2,3,有
x(n+1) = x(n) + step[n] * r(n)
=> e(n+1) = e(n) + step[n] * r(n)
=> e(n+1) = e(n) + (r(n) * r(n)) / (r(n) * A * r(n)) * r(n) #下一步利用公式2，3
=> e(n+1) = e(n) + (r(n) * r(n)) / (lambda * r(n) * r(n)) * (-lambda * e(n))
=> e(n+1) = e(n) - e(n) = 0
```
(证毕)

说明如果能中这样的奖，事情一下子就好办多了。

### 定理3：如果A是对称的且只有一个特征值，那么利用Steepest Descent只需要一步。

```python
# 对于一个对称的矩阵A，我们可以得到n个正交的特征向量，因为改变特征向量的scale对特征值没有影响，
# 所以我们把所有的特征向量单位化，那么有
v(i) * v(j) = 0 if i != j else 1

# 因为n个特征向量是相互正交的，所以它们可以span整个向量空间，所以任意一个残差可以有特征向量的线性组合构成。
e(i) = V * coef # V是特征向量组成的矩阵，coef是每个向量对应的参数
# 于是有
r(i) = -A * e(i) = -lambda * e(i) = -sum([lambda[j] * coef[j] * V[:,j] for j in range(size(coef))])
e(i) * e(i) = sum([coef(j)^2 for j in range(size(coef))])
e(i) * A * e(i) = sum([coef(j)^2 * lambda(j) for j in range(size(coef))])
r(i) * r(i) = sum([coef(j)^2 * lambda(j) ^2])
r(i) * A * r(i) = sum([coef(j)^2 * lambda(j)^3])

# 对于之前的那个公式，有
e(i+1) = e(i) = (r(i) * r(i)) / (r(i) * A * r(i)) * r(i) # 根据上面的替换
=> e(i+1) = e(i) * (sum([coef(j)^2 * lambda(j)^2])) / (sum([coef(j)^2 * lambda(j)^3])) * r(i)

# 如果只有一个特征值，那么lambda(i) == lambda(j),所以
e(i+1) = e(i) + (lambda^2 * sum(coef(j)^2)) / (lambda^3 * sum(coef(j)^2) * (-lambda * e(i))
=> e(i+1) = 0
```

（证毕）

## 一些变化

### Learning rate

在上面的算法里，我们可以求出最好的step，但是有时求最好的step比较复杂，而效果也不一定有多好，所以会直接用一个参数表示步长，如0.1, 0.01等等。

还有的算法里会利用Wolfe定理和line search的方法求步长，都是一些计算量较小的算法。

### Momentum

在最速下降法的过程中，如果初始值不好，很容易走成锯齿状的更新路线，而这样走收敛是非常慢的。所以momentum的作用是相信每一步迭代的潜在价值，在下一步更新时加入上一步的信息，这样能让锯齿状的路线走得快一些。


