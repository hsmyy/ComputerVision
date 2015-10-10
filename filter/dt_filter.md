# Domain Transform Filter

### 问题

这个问题和前面的一些问题类似，我们还是希望去除一些小的细节信息，保留大的主要信息，就像Total Variation, Rolling Guidance Filter里面所做的事情一样。

### 特点

那么这篇paper的特点是什么？它其实希望解决一个问题：之前的一些双通滤波的速度比较慢，无法做到实时处理数据。而本文最大的特点就是通过降维减少数据量。

传统的方法像Bilateral Filter双通滤波，需要用到5维的信息：y,x,r,g,b，而本文希望能够通过降维减少计算量。

### Manifold/流型

paper里面提到，作者希望采用类似joint bilateral filter的方法去做过滤，那么ref_img是什么呢？就是把原图的像素映射到5维空间的一个manifold上。

Manifold这个东西是拓扑学里面比较经典的一个概念。举个例子来说在3维空间下一个卷曲的2维平面就算是一个manifold(不严谨，只做举例)。manifold的一个特点就是它的distance metric和一般欧式空间的距离测度不同，这种感觉就好像游戏里面的迷宫一样，明明你要到的地方就在眼前（欧式空间很近），但是你需要绕很大的圈才能到达（manifold上的距离很远）。

利用流型的模型概念，作者解决了他担心的问题：如何在降维的同时保持原有的距离尺度？降维固然能提高速度，但是如果原本相近的像素在降维后变得远了，是会影响效果的。作者这里说明了利用流型降维是靠谱的。

经过长长的推倒，最终得出了降维的公式：
```python
def manifold(img, sigmaS, sigmaR):
	ctx = array((img.rows, img.cols))
	cty = array((img.rows, img.cols))
	for y in img.rows:
		for x in img.cols:
			ctx[y][x] = ctx[y][x - 1] + 1 + sigmaS / sigmaR *sum(dx(img[y][x][c] for c in channels))

	for x in img.cols:
		for y in img.rows:
			cty[y][x] = cty[y - 1][x] + 1 + sigmaS / sigmaR * sum(dy(img[y][x][c] for c in channels))
	return ctx, cty 
``` 

这里实际上是用到了求曲线长度的积分公式，作者将图像分解成水平方向和垂直方向，把每一行或每一列的梯度想象成一个1维的manifold，然后利用曲线长度积分转换过来

### 2D上的过滤应用


