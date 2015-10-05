# Total Variation/全局变分

### 问题

这次我们遇到的问题是去噪，什么是去噪？去噪就是去除噪声信号。那什么是噪声信号？(好啰嗦啊我都受不了了)

噪声是属于图像中的无意义的信号，甚至是有害的信号，不但不会对分析图像有帮助，而且会产生干扰。于是一些图像处理的第一步就是要去除噪声。

于是问题就变成了两个步骤：

1. 找到噪点

2. 将噪点变换成正常的图像点

往往这两步会合在一起做。

### 噪声的类型

我们常见的噪声大概有两种：

1. 高斯噪声：这种噪声也被成为加性噪声，它可以用一个模型来表示：
```python
def add_gaussian_noise(img):
    model = gaussian_distribution()
    for pixel in img:
        pixel += model.random() 
```
所谓的ori_img就是原始的图片，noise就是叠加在图像上的噪声。一般来说，我们会假设噪声服从高斯分布，这个高斯分布的均值一般为0，方差是某个定值。

2. 椒盐噪声：这种噪声属于替换噪声，它也可以用一个模型来表示：
```python
def add_pepper_salt_noise(img):
    bornouli_pepper = bornouli_model()
    bornouli_salt = bornouli_model()
    for pixel in img:
        if bornouli_pepper.random() == 1:
            pixel = 0
        elif bornouli_salt.random() == 1:
            pixel = 255
        else:
            # keep the same
```

这个名称很形象，椒的颜色是黑色，代表最小的颜色，盐是白色，反之。

知道这两种模型有助于理解噪声的表现形式

### Mean filter/均值过滤器

均值过滤器的伪代码大致如下：
```python
def mean_filter(img, window_size):
    res = [[0] * img.cols] * img.rows;
    for y in range(img.rows):
        for x in range(img.cols):
            ystart = max(0, y - window_size)
            yend = min(img.rows + 1, y + window_size)
            xstart = max(0, x - window_size)
            xend = min(img.cols + 1, x + window_size)
            res[y][x] = sum2d(img, ystart, yend, xstart, xend) / (yend - ystart) / (xend - xstart)            
```

这个问题有更快的解法，可以利用积分图像去解，这里不赘述了。

这个过滤器是一个十分简单的过滤器，他可以很好地解决高斯噪声的问题。因为高斯噪声是一个0均值固定方差的分布，把附近的像素点做平均可以消除掉高斯噪声的方差，这就是这个方法的核心思想。

还有一个类似的方法是中值滤波器，就是求一个窗口内的median来代替当前的像素值。思想也大体类似。

### Edge-aware filter/边缘敏感过滤器

上面的均值滤波器可以解决去除噪声的问题，但是它也带来了一个问题，那就是他会把这个图像变得模糊，图像中的细节也会被清除，这就造成了矛盾。于是乎，前人提出了一个概念，叫做边缘敏感过滤器。

我们需要一种过滤器，它只对细小的细节进行清除，而不会对大的细节进行清除。这样一些核心的细节就得以保留。那么他是怎么做到的呢？

### Total Variation/全局变分法

下面引出这个方法，全局变分。

首先简单地转换全局变分的称呼，我们称呼它为全局梯度。它的实现如下所示：
```python
def tv(img):
    tv_score = 0
    for y in range(0, img.rows - 1):
        for x in range(0, img.cols - 1):
            tv_score += math.abs(img[y][x] - img[y][x+1]) + math.abs(img[y][x] - img[y+1][x]) 
    return tv_score
```

大家可以想象，一副图像如果每一点的像素值都一样，那这幅图象不会表达任何的含义，若是想表达一些含义，那么像素之间必须有像素差，而这些像素值就可以用梯度来表示。所以当这个全局变分很大时，表示图像的像素间差值很大，反之很小。

所以如果我们将一张图像的全局变分变小，那么他的一些细节就会变少。这就是全局变分的第一部分。

全局变分的第二部分叫做fidelity/信任度，可以表示为如下：
```python
def fidelity(img1, img2):
    sum = 0
    for y in range(0, img1.rows):
        for x in range(0, img2.cols):
            sum += (img1[y][x] - img2[y][x]) ^ 2
    return math.sqrt(sum)
```

这个部分是表达了两个图像的相似度，越相似值越小。概念比较简单。

全局变分的理念就是将这两部分融合。已知原图ori_img和噪声图denoise_img，那么全局变分就是要求解下面这个最小化问题：
```python
def obj(ori_img, denoise_img):
    return fidelity(ori_img, denoise_img) + lambda * tv(ori_img)
```

lambda是一个正则项，可作为超参数。

### 全局变分去除掉的是什么？

假设有下面这个1维的图像信号：
```python
denoise_img = [0, 1, 1, 1, 1, 0, 1, 0]
```
可以用上面的公式求得
```python
obj(denoise_img, denoise_img) = 4
```
如果我们找到了下面这个图像信号：
```python
ori_img = [0, 1, 1, 1, 1, 0, 0, 0]
```
我们可以求出上面的目标函数的值(假设lambda=1)：
```python
obj(ori_img, denoise_img) = 1 + 2 = 3
```
可见obj已经变小，这个图像信号得到了优化。

而对于下面这个信号：
```python
ori_img = [0, 1, 1, 1, 0, 0, 1, 0]
obj(ori_img, denoise_img) = 4 + 1 = 5
```

可以看出，如果修改一个较宽的信号，那么他的全局变分不会变少，而对于一个较窄的信号，如果它完全被消除，那么全局变分就会被减少。所以这些较窄的信号就是全局变分的目标，而这些较窄的信号一般也就是噪声或者一些极小的细节信息。

所以，全局变分的核心就是在确保和原图尽可能相近的情况下，减少一些小的细节信息。

关于全局变分的介绍也就到这里了，至于它的解法……呵呵，这个问题有点复杂，有机会再详细介绍。
