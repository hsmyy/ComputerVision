# Domain Transform Filter

### 问题

这个问题和前面的一些问题类似，我们还是希望去除一些小的细节信息，保留大的主要信息，就像Total Variation, Rolling Guidance Filter里面所做的事情一样。

### 特点

那么这篇paper的特点是什么？它其实希望解决一个问题：之前的一些双通滤波的速度比较慢，无法做到实时处理数据。而本文最大的特点就是通过降维减少数据量。

传统的方法像Bilateral Filter双通滤波，需要用到5维的信息：y,x,r,g,b，而本文希望能够通过降维减少计算量。

### Manifold/流型

paper里面提到，作者希望采用类似joint bilateral filter的方法去做过滤，那么ref_img是什么呢？就是把原图的像素映射到5维空间的一个manifold上。

Manifold这个东西是拓扑学里面比较经典的一个概念。举个例子来说在3维空间下一个卷曲的2维平面就算是一个manifold(不严谨，只做举例)。manifold的一个特点就是它的distance metric和一般欧式空间的距离测度不同，这种感觉就好像游戏里面的迷宫一样，明明你要到的地方就在眼前（欧式空间很近），但是你需要绕很大的圈才能到达（manifold上的距离很远）。

前面提到bilateral filter是可以做到edge-preseving的效果的，这个效果来自于原始空间的距离测度。在当前的距离下，利用双边滤波就可以达到应有的效果。因此如果想利用流型做降维，那么就要保证降维后的距离和原始空间的距离保持一致。作者首先考虑的就是把5维降到2维，但是发现除非采用近似算法，不然无法达到这样的效果，而近似算法必然带来精度的降低，于是作者转而探索降维到1维的isometric的方法。

### 1D的降维之路

作者首先将问题转化为2D到1D的domain transform，如何保证转换后的isometric呢？作者经过一定的推导得到了曲线积分的方式，也就是用geodesic metric来替代原始的L1 欧式距离。曲线积分可以很好地保留2D空间下的距离，而且它的单调递增特性也使得新空间的一些基本属性得以保存（metric space的4大特性？）

当然，这样搞完确实是实现了降维，但是原始的双边滤波人家有两个方差参数sigmaS和sigmaR的，你现在丢掉了这个怎么让人家调参数啊？所以作者又开始思考其他的方法。

作者基于一个特性：对于一个卷积操作，把filter的scale扩大1/a倍相当于把signal的scale扩大a倍，所以作者想到了把filter的参数encode到transformed domain的方法：

1）对于signal每一维的信息:i，引入一个参数a(i);

2）将每一维信息的scale扩大a倍

3）应用domain transform

4) 用transform后的kernel进行filter

基于上面这个大发现，和长长的推倒，最终将两个参数嵌入了signal中，得出了降维的公式：
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

这里实际上是用到了求曲线长度的积分公式，作者将图像分解成水平方向和垂直方向，把每一行或每一列的梯度想象成一个1维的manifold，然后利用曲线长度积分转换过来。

### 分析

作者将上面的domain transform公式转换维偏微分方程进行分析sigmaS, sigmaR和patch图像的TV（total variation）与filter kernel的关系：

1) 当sigmaR趋向于无穷大，filter退化成gaussian filter;趋近于0时filter相当于identity;

2）当sigmaS趋近于无穷大，它将变得不起作用，filter的效果将取决于sigmaR和patch signal TV;趋近于0时filter相当于identity;

3）当patch signal的TV比较大，一般认为这是个核心的边缘信息，filter将不对其起作用;趋近于0时，一般认为这是次要边缘或者噪声，filter的威力会剧增。

### 2D上的过滤应用

作者说直接把5D变2D的方法又慢又不好找，于是就想了一个用迭代解决的方法。这个方法前人也用过，就是多做几遍1D的过滤，比方说先做水平，再做垂直，再做水平……经过多轮过滤，效果就像搅拌一样，把所有的像素都拌在了一起。作者在Paper中举了个例子，只要图像不像迷宫一样复杂，一般的纹理即使有绕弯，经过几轮也就找到了。

当然，这时候需要注意一个问题，就是随着过滤迭代不断进行，其中的方差参数需要随着调节，作者也给出了解决方案。

最后，作者也做了收敛性测试，实验表明3轮迭代基本就可以收敛了。

### 具体的1D过滤器

架子搭好了，下面就是具体的算法了，作者在paper中讲了3个算法:NC，IC，RF。实话说除了第一个其他的都不太理解。做实验测试发现RF的速度奇快。作者还把这三个算法分别和其他的一些经典算法做比较，发现了他们与经典算法的相似性。

### 应用场景

我觉得这篇文章的这一部分也是很吸引人，他详细讲述了过滤器的各种玩法。比方说细节增强，将每一轮迭代所损失的细节保存下来，可以做不同粒度的增强；还有铅笔画，利用filter的normalization value做想素值。这些都极大地扩展了filter的应用。

### 总结

### Reference

[Paper](http://www.inf.ufrgs.br/~eslgastal/DomainTransform/Gastal_Oliveira_SIGGRAPH2011_Domain_Transform.pdf)

[Slide](http://vis.berkeley.edu/courses/cs294-69-fa11/wiki/images/3/3a/Domain_Transform_Slides.pdf)

