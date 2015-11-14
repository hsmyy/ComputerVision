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

def vecDotProduct2(a, b, angle):
  return vecNorm(a) * vecNorm(b) * cos(angle)
```
对于第二种方法，可以通过余弦定理推导：三角形三边a,b,c，c^2=a^2+b^2-2abcos(angle)
