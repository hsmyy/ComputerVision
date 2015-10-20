# 网络与训练预测

### 完整的例子

上一节主要分析了网络每一层，下面看看他们是如何组合起来的。首先是main.cpp的完整例子：

```c++
// 定义网络
network<mse, gradient_descent_levenberg_marquardt> nn;
// 构建网络
nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6) // 32x32 in, 5x5 kernel, 1-6 fmaps conv
       << average_pooling_layer<tan_h>(28, 28, 6, 2) // 28x28 in, 6 fmaps, 2x2 subsampling
       << convolutional_layer<tan_h>(14, 14, 5, 6, 16,
                                     connection_table(connection, 6, 16)) // with connection-table
       << average_pooling_layer<tan_h>(10, 10, 16, 2)
       << convolutional_layer<tan_h>(5, 5, 5, 16, 120)
       << fully_connected_layer<tan_h>(120, 10);
// 设定minibatch参数
int minibatch_size = 10;
nn.optimizer().alpha *= std::sqrt(minibatch_size);
// 省略其中的hook函数
// 训练
nn.train(train_images, train_labels, minibatch_size, 20, on_enumerate_minibatch, on_enumerate_epoch);
// 测试
nn.test(test_images, test_labels).print_detail(std::cout);
```

### 训练
训练的核心代码：
```c++
for (int iter = 0; iter < epoch; iter++) {
    if (optimizer_.requires_hessian())
        calc_hessian(in);
    for (size_t i = 0; i < in.size(); i+=batch_size) {
        train_once(&in[i], &t[i], std::min(batch_size, in.size() - i));
        on_batch_enumerate();
    }
    on_epoch_enumerate();
}
// 单个minibatch的训练
void train_once(const vec_t* in, const vec_t* t, int size) {
  bprop(fprop(in), t); //做完前向做后向
  layers_.update_weights(&optimizer_, 1, 1); // 更新权重
}
```

整体的架构是不是还算简单？看到这里实际上我们已经完成了对最基本的tiny_cnn的了解了，但是我们想要的显然没这么简单。我们看到代码中有很多关于并行优化的问题，下面的问题就是tiny_cnn如何实现并行训练的？

### 并行训练

