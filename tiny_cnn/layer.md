# 网络层

### 层的构造

layer_base作为层的base类，他定义了许多基本的属性：

in_size_, out_size_:

W_, b_:

dW_, db_:

Whessian_, bhessian:

output_, prev_delta:

layer类基本上打了酱油。

关于每一层的定义可以看main.cpp里面的定义：

```c++
nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6) // 32x32 in, 5x5 kernel, 1-6 fmaps conv
       << average_pooling_layer<tan_h>(28, 28, 6, 2) // 28x28 in, 6 fmaps, 2x2 subsampling
       << convolutional_layer<tan_h>(14, 14, 5, 6, 16,
                                     connection_table(connection, 6, 16)) // with connection-table
       << average_pooling_layer<tan_h>(10, 10, 16, 2)
       << convolutional_layer<tan_h>(5, 5, 5, 16, 120)
       << fully_connected_layer<tan_h>(120, 10);
```

以下是layer_base的构造函数：
```c++
layer_base(layer_size_t in_dim, layer_size_t out_dim, size_t weight_dim, size_t bias_dim);
```
而几个具体层的构造函数调用父类构造函数的方法如下所示：

全连接层：
```c++
fully_connected_layer(
       layer_size_t in_dim, //只需要输入输出的维度
       layer_size_t out_dim)
        : layer<Activation>(
        in_dim, 
        out_dim, 
        size_t(in_dim) * out_dim, 
        out_dim
        ), filter_(out_dim) {}
```

这个和神经网络的结构一样，很简单。

卷积层：

```c++
convolutional_layer(
layer_size_t in_width,  //对应上面例子中的第一层，=32
layer_size_t in_height, //=32
layer_size_t window_size, //=5
layer_size_t in_channels, //=1
layer_size_t out_channels) //6
    : partial_connected_layer<Activation>(
    in_width * in_height * in_channels, //对应layer_base的in_dim
    (in_width - window_size + 1) * (in_height - window_size + 1) * out_channels,  //out_dim
    sqr(window_size) * in_channels * out_channels, //weight_dim
    out_channels) // bias_dim
```

可以看出卷积层的计算方法。对于权重核，每一对channel都有一个自己的权重核

Max pooling层：

```c++
max_pooling_layer(
       layer_size_t in_width, 
       layer_size_t in_height, 
       layer_size_t in_channels, 
       layer_size_t pooling_size)
        : layer<Activation>(
        in_width * in_height * in_channels, //in_dim
        in_width * in_height * in_channels / sqr(pooling_size), //out_dim
        0, 0), // weight_dim & bias_dim
```

可以看出Pooling层没有权重。

### 层的核心函数

每一层的核心函数有下面几个：

```c++
virtual activation::function& activation_function() = 0; // 获取激活函数
virtual const vec_t& forward_propagation(const vec_t& in, size_t worker_index) = 0; //前向传播
virtual const vec_t& back_propagation(const vec_t& current_delta, size_t worker_index) = 0; //后向传播残差
virtual const vec_t& back_propagation_2nd(const vec_t& current_delta2) = 0; // 后向传播2阶hessian
```

### 前向传播

这里用到了util.h里面的模板，这里按无TBB,OMP去分析

全连接层

```c++
for_(this->parallelize_, 0, this->out_size_, [&](const blocked_range& r) {
     for (int i = r.begin(); i < r.end(); i++) { // 对于每一个输出层的值
         float_t z = 0.0;
         for (int c = 0; c < this->in_size_; c++) // 等于输入层的每一个值和权重的乘积再加和
             z += this->W_[c*this->out_size_ + i] * in[c];

         z += this->b_[i]; // 加上偏置项
         this->output_[index][i] = this->a_.f(z); // 在index处保存output
     }
 });
```

卷积层的代码来自于partial connectioned layer

```c++
// 网络启动前余弦算法坐标对应的方法

typedef std::vector<std::pair<unsigned short, unsigned short> > wi_connections;
std::vector<wi_connections> out2wi_; // out_id -> [(weight_id, in_id)]
std::vector<size_t> out2bias_;

for_(this->parallelize_, 0, this->out_size_, [&](const blocked_range& r) {
     for (int i = r.begin(); i < r.end(); i++) { // 对于输出层的每一个值
         const wi_connections& connections = out2wi_[i]; //找到其对应的输入层和权重层的映射
         float_t a = 0.0;

         for (auto connection : connections)// 然后是乘加
             a += this->W_[connection.first] * in[connection.second]; //

         a *= scale_factor_;
         a += this->b_[out2bias_[i]]; // 乘以权重，加上bias
         this->output_[index][i] = this->a_.f(a); // 9.6%
     }
 });
```

max_pooling_layer:

```c++
// 预先算好的映射
std::vector<std::vector<int> > out2in_; // mapping out => in (1:N) 保存映射
std::vector<int> out2inmax_; // mapping out => max_index(in) (1:1) 保存最大值位置

virtual const vec_t& forward_propagation(const vec_t& in, size_t index) {
 for_(this->parallelize_, 0, this->out_size_, [&](const blocked_range& r) {
     for (int i = r.begin(); i < r.end(); i++) { // 对每一个输出层
         const auto& in_index = out2in_[i]; // 找到映射
         float_t max_value = std::numeric_limits<float_t>::lowest();
         
         for (auto j : in_index) { // 找出最大的，并保存最大的位置
             if (in[j] > max_value) {
                 max_value = in[j];
                 out2inmax_[i] = j;
             }
         }
         this->output_[index][i] = max_value;
     }
 });
}
```

### 反向传播

首先全连接层

```c++
const vec_t& back_propagation(const vec_t& current_delta, size_t index) {
 const vec_t& curr_delta = this->filter_.filter_bprop(current_delta, index); // 默认返回current_delta本身
 const vec_t& prev_out = this->prev_->output(index);
 const activation::function& prev_h = this->prev_->activation_function();
 vec_t& prev_delta = this->prev_delta_[index];
 vec_t& dW = this->dW_[index];
 vec_t& db = this->db_[index];

 // 这个循环用来计算上一层的残差，也就是sum(本层和上层的权重*本层残差)*上层激活函数的导数
 for (int c = 0; c < this->in_size_; c++) {
     // propagate delta to previous layer
     // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
     prev_delta[c] = vectorize::dot(&curr_delta[0], &this->W_[c*this->out_size_], this->out_size_);
     prev_delta[c] *= prev_h.df(prev_out[c]);
 }
 // 这个循环把本层的权重更新进行计算
 for_(this->parallelize_, 0, this->out_size_, [&](const blocked_range& r) {
     // accumulate weight-step using delta
     // dW[c * out_size + i] += current_delta[i] * prev_out[c]
     for (int c = 0; c < this->in_size_; c++)
         vectorize::muladd(&curr_delta[0], prev_out[c], r.end() - r.begin(), &dW[c*this->out_size_ + r.begin()]);

     for (int i = r.begin(); i < r.end(); i++) 
         db[i] += curr_delta[i];
 });
}
```

然后是卷积层
```c++
virtual const vec_t& back_propagation(const vec_t& current_delta, size_t index) {
 const vec_t& prev_out = this->prev_->output(index);
 const activation::function& prev_h = this->prev_->activation_function();
 vec_t& prev_delta = this->prev_delta_[index];

 // 计算上一层的残差
 for_(this->parallelize_, 0, this->in_size_, [&](const blocked_range& r) {
     for (int i = r.begin(); i != r.end(); i++) {
         const wo_connections& connections = in2wo_[i];
         float_t delta = 0.0;

         for (auto connection : connections) 
             delta += this->W_[connection.first] * current_delta[connection.second]; // 40.6%

         prev_delta[i] = delta * scale_factor_ * prev_h.df(prev_out[i]); // 2.1%
     }
 });
 // 分别更新w和b的权重
 for_(this->parallelize_, 0, weight2io_.size(), [&](const blocked_range& r) {
     for (int i = r.begin(); i < r.end(); i++) {
         const io_connections& connections = weight2io_[i];
         float_t diff = 0.0;

         for (auto connection : connections) // 11.9%
             diff += prev_out[connection.first] * current_delta[connection.second];

         this->dW_[index][i] += diff * scale_factor_;
     }
 });

 for (size_t i = 0; i < bias2out_.size(); i++) {
     const std::vector<layer_size_t>& outs = bias2out_[i];
     float_t diff = 0.0;

     for (auto o : outs)
         diff += current_delta[o];    

     this->db_[index][i] += diff;
 } 
}
```

最后是max_pooling:
```c++
virtual const vec_t& back_propagation(const vec_t& current_delta, size_t index) {
 const vec_t& prev_out = this->prev_->output(index);
 const activation::function& prev_h = this->prev_->activation_function();
 vec_t& prev_delta = this->prev_delta_[index];
 // 这里主要是做残差的传导
 for_(this->parallelize_, 0, this->in_size_, [&](const blocked_range& r) {
     for (int i = r.begin(); i != r.end(); i++) {
         int outi = in2out_[i];
         prev_delta[i] = (out2inmax_[outi] == i) ? current_delta[outi] * prev_h.df(prev_out[i]) : 0.0;
     }
 });
 return this->prev_->back_propagation(this->prev_delta_[index], index);
}
```

具体的公式请参阅其他的文章，这里不做介绍

