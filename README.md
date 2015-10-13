# Computer Vision
不好意思，接下来都是中文了……
作为从事机器视觉已近一年的我来说，从最初的啥都不懂，到现在对一些问题能够有一定的想法，可以说还是有了很大的提高，希望在这个平台能够多分享一些对机器视觉问题的理解。

作为一个CS背景的人，在学习CV的路上遇到的一个很大的问题是————很多教材和博客的讲解充满了数学的概念但是最终没有落实到代码上，这对于CS专业的人来说实在是不过瘾，一个积分公式怎么能让我看清背后的本质？（大侠你能不能写成for循环……）

我想尽量站在CS的角度去描述每一个问题和算法，多帖代码少写公式，让CS的高手快速领悟CV的真谛。

## 目录

### Line detection/直线检测

#### [Hough Transformation/霍夫变换](line/hough.md)
#### [LSD/线段检测](line/lsd.md)

### Edge-aware Filter

#### [Total Variation/全局变分](filter/tv.md)
#### [Rolling Guidance Filter](filter/rolling.md)
#### [Domain Transfom Filter](filter/dt_filter.md)
#### [Guided Filter](filter/guided.md)

### Filter/过滤

#### [Decolor/灰度化](filter/decolor.md)
#### [Fourier Transformation/傅里叶变换](filter/fourier.md)
#### [Wavelet Transformation/小波变换](filter/wavelet.md)

### Segmentation/分割

#### [Otsu&Niblack/二值化](seg/niblack.md)

### Morphology/形态学

#### [Skeleton Extraction/骨架提取](mor/skeleton.md)

### Feature/特征抽取

#### [Shape Context/形状上下文](feature/shapeContext.md)
#### [Sift](feature/sift.md)
#### [Hog](feature/hog.md)
#### [LBP](feature/lbp.md)
