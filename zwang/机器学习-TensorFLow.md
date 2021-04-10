## TensorFlow

* Python 列表(list):

  可以存储不同数据类型；在内存中不连续存放；读写效率低，占用内存空间大；不适合做数值计算。

* Numpy 数组(ndarray):

  元素数据类型相同; 在内存中连续存放；读写效率高，存储空间小；在CPU中运算，不能主动监测、利用GPU进行运算。

* TensorFlow张量(Tensor):

  可以高速运行于GPU和TPU之上；支持CPU、嵌入式、等多种环境。

TensorFlow 的基本运算、参数命名、运算规则、API设计与Numpy非常相似。

**创建张量**

张量由Tensor类实现，每个张量都是一个Tensor对象。

创建：`tf.constant()` 函数

`tf.constant(value, dtype, shape)`

value: 数字／Python列表／Numpy数组

dtype: 元素的数据类型

shape: 张量的维度

TensorFlow的张量是对Numpy数组的封装，可以用过`张量.numpy()`方法访问封装的numpy数组。

```python
a = tf.constant([[1, 2], [3, 4]])
print(a.numpy())
```

