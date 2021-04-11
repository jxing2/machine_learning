```python
import tensorflow as tf

print(tf.__version__)
tf.executing_eagerly()
```

    2.5.0-rc0





    True



# TensorFlow

* Python 列表(list):

  可以存储不同数据类型；在内存中不连续存放；读写效率低，占用内存空间大；不适合做数值计算。

* Numpy 数组(ndarray):

  元素数据类型相同; 在内存中连续存放；读写效率高，存储空间小；在CPU中运算，不能主动监测、利用GPU进行运算。

* TensorFlow张量(Tensor):

  可以高速运行于GPU和TPU之上；支持CPU、嵌入式、等多种环境。

TensorFlow 的基本运算、参数命名、运算规则、API设计与Numpy非常相似。TensorFlow的所有运算都是在张量之间运行的，Numpy数组仅作为输入输出。
在CPU环境下，张量和Numpy数组共享同一段内存。多维张量在内存中是以一维数组的方式连续存储的。

## 创建张量

张量由Tensor类实现，每个张量都是一个Tensor对象。

1. `tf.constant()` 函数

`tf.constant(value, dtype, shape)`

value: 数字／Python列表／Numpy数组 ／ bool型

dtype: 元素的数据类型

shape: 张量的维度


```python
# 创建张量，参数为Python列表
tf.constant([[1, 2], [3, 4]])
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[1, 2],
           [3, 4]], dtype=int32)>



TensorFlow的张量是对Numpy数组的封装，可以用过`张量.numpy()`方法访问封装的numpy数组。


```python
a = tf.constant([[1, 2], [3, 4]])
a.numpy()
```




    array([[1, 2],
           [3, 4]], dtype=int32)




```python
type(a)
```




    tensorflow.python.framework.ops.EagerTensor



表示a是一个Eager模式下的张量


```python
print(a)
```

    tf.Tensor(
    [[1 2]
     [3 4]], shape=(2, 2), dtype=int32)



```python
# 创建张量，参数是数值
tf.constant(1, dtype=tf.float64)
```




    <tf.Tensor: shape=(), dtype=float64, numpy=1.0>




```python
tf.constant(1)
```




    <tf.Tensor: shape=(), dtype=int32, numpy=1>




```python
# 创建张量，参数是numpy array
import numpy as np

tf.constant(np.array([1, 2]))
```




    <tf.Tensor: shape=(2,), dtype=int64, numpy=array([1, 2])>



numpy数组创建的张量默认的数据类型来源于numpy数组的数据类型，由于numpy默认是64位，所以这里是64位，但实际上32位足够使用了也比较快，建议使用32位：


```python
tf.constant(np.array([1, 2]), dtype=tf.int32)
```




    <tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2], dtype=int32)>



`tf.cast(x, dtype)` 改变张量中元素的数据类型，但是注意一般是从低维度数据类型到高纬度数据类型转换，不然从高纬度向低纬度转换容易溢出。


```python
a = tf.constant(np.array([1, 2]))
b = tf.cast(a, dtype=tf.int32)
print(a.dtype)
print(b.dtype)
```

    <dtype: 'int64'>
    <dtype: 'int32'>



```python
# 创建张量，参数是bool型
a = tf.constant(True)
a
```




    <tf.Tensor: shape=(), dtype=bool, numpy=True>




```python
# bool型与整型之间可以互相转换
b = tf.cast(a, tf.int32)
b
```




    <tf.Tensor: shape=(), dtype=int32, numpy=1>




```python
# 非0转换为True，0转换为False
c = tf.constant([1, -1, 0, 2, 3])
d = tf.cast(c, dtype=tf.bool)
d
```




    <tf.Tensor: shape=(5,), dtype=bool, numpy=array([ True,  True, False,  True,  True])>




```python
# 创建张量，参数是string
a = tf.constant("hello")
a
```




    <tf.Tensor: shape=(), dtype=string, numpy=b'hello'>



在python3中，string是用unicode编码的，到numpy中要转换为字节串，字符串前面需要加上 `b` 

2. `tf.convert_to_tensor(numpy数组/python列表/bool型／字符串)` 创建tensor


```python
na = np.arange(12).reshape(3, 4)
ta = tf.convert_to_tensor(na)
ta
```




    <tf.Tensor: shape=(3, 4), dtype=int64, numpy=
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])>



3. `tf.is_tensor()` 判断是否张量类型


```python
tf.is_tensor(na)
```




    False




```python
tf.is_tensor(ta)
```




    True



也可以用python中的isinstance()判断类型


```python
isinstance(na, np.ndarray)
```




    True




```python
isinstance(ta, tf.Tensor)
```




    True



4. 创建全0或全1张量 `tf.zeros(shape, dtype=tf.float32)` `tf.ones(shape, dtype=tf.float32)` 默认数据类型是tf.float32


```python
tf.zeros(shape=(2, 1))
```




    <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
    array([[0.],
           [0.]], dtype=float32)>




```python
tf.ones(shape=(2, 2))
```




    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[1., 1.],
           [1., 1.]], dtype=float32)>




```python
tf.ones([3, 4])
```




    <tf.Tensor: shape=(3, 4), dtype=float32, numpy=
    array([[1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.]], dtype=float32)>



5. 创建所有元素值都相同的张量 `tf.fill(dims, value)`


```python
tf.fill(dims=(2, 2), value=5)
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[5, 5],
           [5, 5]], dtype=int32)>




```python
tf.fill([2, 2], 5)
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[5, 5],
           [5, 5]], dtype=int32)>




```python
tf.constant(value=5, shape=[2, 2])
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[5, 5],
           [5, 5]], dtype=int32)>




```python
tf.constant(5, shape=[2, 2])
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[5, 5],
           [5, 5]], dtype=int32)>



为避免混淆不同形式，尽量使用关键词形式

6. 创建服从标准正态分布的张量 `tf.random.normal(shape, mean, stddev, dtype)` 默认为标准正态分布，即mean=0, stddev=1


```python
tf.random.normal([3, 3, 3], mean=0, stddev=2)
```




    <tf.Tensor: shape=(3, 3, 3), dtype=float32, numpy=
    array([[[-0.6749023 , -3.2487562 ,  6.4994273 ],
            [-1.271746  ,  2.7861636 , -1.6927681 ],
            [ 3.5853    ,  2.5916934 , -2.175253  ]],
    
           [[ 1.2934587 , -4.643426  ,  0.06734839],
            [ 4.158525  , -0.37307698,  1.8560337 ],
            [ 0.70524436, -2.9329398 ,  4.969238  ]],
    
           [[-3.6922245 , -0.2457193 ,  1.3220365 ],
            [-2.2036026 ,  1.9504955 ,  2.171845  ],
            [-2.882198  ,  3.1456118 ,  0.23984219]]], dtype=float32)>



创建截断正态分布的张量，返回一个截断的正态分布，截断的标准是2倍的标准差 `tf.random.truncated_normal(shape, mean, stddev, dtype)`

例如：当均值为0， 标准差为1时：

使用`tf.random.truncated_normal()`不可能出现[-2, 2]以外的点

使用`tf.random.normal()`可能出现[-2, 2]以外的点

`tf.random.set(value)` 设置随机种子，可以产生同样的随机张量。


```python
tf.random.set_seed(10)
tf.random.normal(shape=(2, 2))
```




    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[-0.8757808 ,  0.3356369 ],
           [-0.35219625, -0.3031456 ]], dtype=float32)>




```python
tf.random.set_seed(10)
tf.random.normal(shape=(2, 2))
```




    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[-0.8757808 ,  0.3356369 ],
           [-0.35219625, -0.3031456 ]], dtype=float32)>



7. 创建服从均匀分布的张量 `tf.random.uniform(shape, minval, maxval, dtype)`

minval: 最小值

maxval: 最大值

区间前闭后开，不包括最大值。


```python
tf.random.uniform(shape=(2, 3), minval=0, maxval=10)
```




    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[5.5478153, 4.2658195, 1.4140713],
           [7.0369396, 3.726666 , 6.976985 ]], dtype=float32)>



8. 创建随机排序的张量  `tf.random.shuffle()`, 注意打乱的是第一维的顺序


```python
a = tf.constant([[1, 2], [3, 4], [5, 6]])
tf.random.shuffle(a)
```




    <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
    array([[3, 4],
           [1, 2],
           [5, 6]], dtype=int32)>




```python
a = np.arange(10)
tf.random.shuffle(a)
```




    <tf.Tensor: shape=(10,), dtype=int64, numpy=array([7, 8, 4, 5, 9, 1, 0, 2, 6, 3])>



9. 创建序列 `tf.range(start, limit, delta=1, dtype)`，前开后闭，与np.arange()类似


```python
# 创建序列
tf.range(0, 10)
```




    <tf.Tensor: shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)>




```python
#创建偶数序列
tf.range(0, 10, delta=2)
```




    <tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 2, 4, 6, 8], dtype=int32)>




```python
#创建奇数序列
tf.range(1, 10, delta=2)
```




    <tf.Tensor: shape=(5,), dtype=int32, numpy=array([1, 3, 5, 7, 9], dtype=int32)>



Tensor对象的属性: ndim, shape, dtype


```python
a = tf.random.normal(shape=(2, 2))
a.ndim
```




    2




```python
a.shape
```




    TensorShape([2, 2])




```python
a.dtype
```




    tf.float32



也可以用`tf.shape(TensorObj)`, `tf.size(TensorObj)`, `tf.rank(TensorObj)`函数形式分别获取形状、元素个数、维度


```python
tf.shape(a)
```




    <tf.Tensor: shape=(2,), dtype=int32, numpy=array([2, 2], dtype=int32)>




```python
tf.size(a)
```




    <tf.Tensor: shape=(), dtype=int32, numpy=4>




```python
tf.rank(a)
```




    <tf.Tensor: shape=(), dtype=int32, numpy=2>



## 张量维度变换

`tf.reshape(tensor, shape)`



```python
# 方法一
import tensorflow as tf

a= tf.range(24)
tf.reshape(a, [2, 3, 4])
```




    <tf.Tensor: shape=(2, 3, 4), dtype=int32, numpy=
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]], dtype=int32)>




```python
# 方法二
import numpy as np

tf.constant(np.arange(24).reshape(2, 3, 4))
```




    <tf.Tensor: shape=(2, 3, 4), dtype=int64, numpy=
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])>



`tf.reshape(tensor, shape)` 中的shape也可以为-1，即表示不知道这个维度的数值，tf可以自己根据其他维度推导出来。


```python
a = tf.range(24)
tf.reshape(a, [2, 3, -1])
```




    <tf.Tensor: shape=(2, 3, 4), dtype=int32, numpy=
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]], dtype=int32)>



tf.reshape转换只是改变了视图（逻辑顺序），内存中的张量顺序并没有改变


```python
a
```




    <tf.Tensor: shape=(24,), dtype=int32, numpy=
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23], dtype=int32)>



## 张量维度增删

`tf.expand_dims(input, axis)`

input：输入张量

axis: 增加的轴方向

增加后的轴的维度为1

`tf.squeeze(input, axis)`

只能删除长度为1的维度，所以是squeeze

增加和删除维度都不改变张量的存储，只是改变了视图。


```python
a = tf.constant(np.arange(10))
a.shape
```




    TensorShape([10])




```python
a
```




    <tf.Tensor: shape=(10,), dtype=int64, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])>




```python
# 在axis=1的方向上增加维度
a = tf.constant(np.arange(10))
a = tf.expand_dims(a, 1)
a.shape
```




    TensorShape([10, 1])




```python
a
```




    <tf.Tensor: shape=(10, 1), dtype=int64, numpy=
    array([[0],
           [1],
           [2],
           [3],
           [4],
           [5],
           [6],
           [7],
           [8],
           [9]])>




```python
# 在axis=0的维度上增加维度
a = tf.constant(np.arange(10))
a = tf.expand_dims(a, 0)
a.shape
```




    TensorShape([1, 10])




```python
a
```




    <tf.Tensor: shape=(1, 10), dtype=int64, numpy=array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])>



## 张量转置

`tf.transpose(input, perm)`

perm: 轴的顺序，比如[0, 1], [1, 0]

转置不仅改变了张量的视图，也改变了张量的顺序


```python
a = tf.constant(np.arange(12).reshape(3, 4))
a
```




    <tf.Tensor: shape=(3, 4), dtype=int64, numpy=
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])>




```python
a = tf.constant(np.arange(12).reshape(3, 4))
a = tf.transpose(a, [0, 1]) #轴的顺序没有变
a
```




    <tf.Tensor: shape=(3, 4), dtype=int64, numpy=
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])>




```python
a = tf.constant(np.arange(12).reshape(3, 4))
a = tf.transpose(a, [1, 0])
a
```




    <tf.Tensor: shape=(4, 3), dtype=int64, numpy=
    array([[ 0,  4,  8],
           [ 1,  5,  9],
           [ 2,  6, 10],
           [ 3,  7, 11]])>




```python
a = tf.constant(np.arange(12).reshape(3, 4))
a = tf.transpose(a)
a
```




    <tf.Tensor: shape=(4, 3), dtype=int64, numpy=
    array([[ 0,  4,  8],
           [ 1,  5,  9],
           [ 2,  6, 10],
           [ 3,  7, 11]])>



## 张量拼接和分割

`tf.concat(tensors, axis)`

tensors: 张量列表

axis: 拼接的轴

拼接并不会增加新的维度

拼接和分割改变了张量的视图，但张量的存储顺序并没有改变。


```python
a = tf.reshape(tf.range(6), [2, 3])
b = tf.reshape(tf.range(10, 16), [2, 3])
print(a, b)
```

    tf.Tensor(
    [[0 1 2]
     [3 4 5]], shape=(2, 3), dtype=int32) tf.Tensor(
    [[10 11 12]
     [13 14 15]], shape=(2, 3), dtype=int32)



```python
tf.concat([a, b], 0) # 可以看到轴0上的长度增加了
```




    <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [10, 11, 12],
           [13, 14, 15]], dtype=int32)>




```python
tf.concat([a, b], 1) # 可以看到轴1上的长度增加了
```




    <tf.Tensor: shape=(2, 6), dtype=int32, numpy=
    array([[ 0,  1,  2, 10, 11, 12],
           [ 3,  4,  5, 13, 14, 15]], dtype=int32)>




```python
tf.concat([a, b], -1) # 可以看到最后一个轴上的长度增加了
```




    <tf.Tensor: shape=(2, 6), dtype=int32, numpy=
    array([[ 0,  1,  2, 10, 11, 12],
           [ 3,  4,  5, 13, 14, 15]], dtype=int32)>



`tf.split(value, num_or_size_splits, axis=0)`

value: 待分割的张量

num_or_size_splits: 如果是一个数值，就是分为几份；如果是一个列表，则是每份的维度长度

axis: 分割的轴方向


```python
a = tf.reshape(tf.range(24), [4, 6])
a
```




    <tf.Tensor: shape=(4, 6), dtype=int32, numpy=
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23]], dtype=int32)>




```python
a = tf.reshape(tf.range(24), [4, 6])
tf.split(a, 2, 0)
```




    [<tf.Tensor: shape=(2, 6), dtype=int32, numpy=
     array([[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11]], dtype=int32)>,
     <tf.Tensor: shape=(2, 6), dtype=int32, numpy=
     array([[12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23]], dtype=int32)>]




```python
a = tf.reshape(tf.range(24), [4, 6])
tf.split(a, 2, 1)
```




    [<tf.Tensor: shape=(4, 3), dtype=int32, numpy=
     array([[ 0,  1,  2],
            [ 6,  7,  8],
            [12, 13, 14],
            [18, 19, 20]], dtype=int32)>,
     <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
     array([[ 3,  4,  5],
            [ 9, 10, 11],
            [15, 16, 17],
            [21, 22, 23]], dtype=int32)>]




```python
a = tf.reshape(tf.range(24), [4, 6])
tf.split(a, [2, 2, 2], 1)
```




    [<tf.Tensor: shape=(4, 2), dtype=int32, numpy=
     array([[ 0,  1],
            [ 6,  7],
            [12, 13],
            [18, 19]], dtype=int32)>,
     <tf.Tensor: shape=(4, 2), dtype=int32, numpy=
     array([[ 2,  3],
            [ 8,  9],
            [14, 15],
            [20, 21]], dtype=int32)>,
     <tf.Tensor: shape=(4, 2), dtype=int32, numpy=
     array([[ 4,  5],
            [10, 11],
            [16, 17],
            [22, 23]], dtype=int32)>]



## 张量堆叠和分解

`tf.stack(values, axis)`

values: 待堆叠的张量

axis: 堆叠方向

堆叠和分解会改变维度


```python
a = tf.range(6)
b = tf.range(10, 16)
tf.stack([a, b], 0)
```




    <tf.Tensor: shape=(2, 6), dtype=int32, numpy=
    array([[ 0,  1,  2,  3,  4,  5],
           [10, 11, 12, 13, 14, 15]], dtype=int32)>




```python
a = tf.range(6)
b = tf.range(10, 16)
tf.stack([a, b], 1)
```




    <tf.Tensor: shape=(6, 2), dtype=int32, numpy=
    array([[ 0, 10],
           [ 1, 11],
           [ 2, 12],
           [ 3, 13],
           [ 4, 14],
           [ 5, 15]], dtype=int32)>



`tf.unstack(values, axis)`

values: 待分解的张量

axis: 分解的轴方向

张量分解是张量堆叠的逆运算，分解后的张量都比原张量少了1维


```python
a = tf.reshape(tf.range(12), [3, 4])
a
```




    <tf.Tensor: shape=(3, 4), dtype=int32, numpy=
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=int32)>




```python
a = tf.reshape(tf.range(12), [3, 4])
tf.unstack(a, axis=0)
```




    [<tf.Tensor: shape=(4,), dtype=int32, numpy=array([0, 1, 2, 3], dtype=int32)>,
     <tf.Tensor: shape=(4,), dtype=int32, numpy=array([4, 5, 6, 7], dtype=int32)>,
     <tf.Tensor: shape=(4,), dtype=int32, numpy=array([ 8,  9, 10, 11], dtype=int32)>]




```python
a = tf.reshape(tf.range(12), [3, 4])
tf.unstack(a, axis=1)
```




    [<tf.Tensor: shape=(3,), dtype=int32, numpy=array([0, 4, 8], dtype=int32)>,
     <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 5, 9], dtype=int32)>,
     <tf.Tensor: shape=(3,), dtype=int32, numpy=array([ 2,  6, 10], dtype=int32)>,
     <tf.Tensor: shape=(3,), dtype=int32, numpy=array([ 3,  7, 11], dtype=int32)>]



## 张量索引和切片

张量索引和切片的方法与numpy数组几乎完全一样

索引：

* 一维张量

a[1]

* 二维张量

b[1, 1]

b[1][1]

* 三维张量

c[1, 1, 1]

c[1][1][1]

切片：

a(start:end:step) 前闭后开，三个数都可以省略


```python
a = tf.range(10)
a[::]
```




    <tf.Tensor: shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)>




```python
a = tf.range(10)
a[1::2]
```




    <tf.Tensor: shape=(5,), dtype=int32, numpy=array([1, 3, 5, 7, 9], dtype=int32)>




```python
a = tf.range(10)
a[::-2]
```




    <tf.Tensor: shape=(5,), dtype=int32, numpy=array([9, 7, 5, 3, 1], dtype=int32)>



二维切片时，维度之间用`,`隔开

```
#假设我们有一个2维数据
import numpy as np
import pandas as pd
import tensorflow as tf

df_data = pd.read_csv("data.csv")
np_data = np.array(df_data)
data = tf.convert_to_tensor(np_data)

#读取第一个样本的所有列
data[0, :]

#读取前5个样本的前4列
data[0:5, 0:4]

#读取所有样本的第一列
data[:, 0]
```

3维张量切片则为

`data[0:10, :, :]` 比如为前10个图片的所有像素

## 张量数据提取

`tf.gather(params, indices)` 用一个索引列表，将给定张量中对应索引值的元素提取出来

params: 给定张量

indices：索引列表


```python
import tensorflow as tf

a = tf.range(6)
tf.gather(a, indices=[0, 2, 4])
```

    INFO:tensorflow:Enabling eager execution
    INFO:tensorflow:Enabling v2 tensorshape
    INFO:tensorflow:Enabling resource variables
    INFO:tensorflow:Enabling tensor equality
    INFO:tensorflow:Enabling control flow v2





    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([0, 2, 4], dtype=int32)>



对多维张量提取，`tf.gather(params, axis, indices)`

axis: 提取的轴方向


```python
a = tf.reshape(tf.range(25), [5, 5])
a
```




    <tf.Tensor: shape=(5, 5), dtype=int32, numpy=
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]], dtype=int32)>




```python
tf.gather(a, axis=0, indices=[0, 2, 3])
```




    <tf.Tensor: shape=(3, 5), dtype=int32, numpy=
    array([[ 0,  1,  2,  3,  4],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]], dtype=int32)>




```python
tf.gather(a, axis=1, indices=[0, 2, 3])
```




    <tf.Tensor: shape=(5, 3), dtype=int32, numpy=
    array([[ 0,  2,  3],
           [ 5,  7,  8],
           [10, 12, 13],
           [15, 17, 18],
           [20, 22, 23]], dtype=int32)>



`tf.gather()`只能对一个维度索引，而`tf.gather_nd()`能同时对多个维度索引


```python
tf.gather_nd(a, [[0, 0], [1, 1], [2, 2]])
```




    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([ 0,  6, 12], dtype=int32)>



```
#如果a是一个3维的图像集数据
tf.gather_nd(a, [[0], [1]], [2]) 代表前3个图像
```

## 张量运算

张量运算的这些函数接口不统一，有的在math模块下，最好涉及数学运算的都加上math. 张量运算要求各个张量的元素数据类型必须一致.

**加减乘除**



| 算术操作          | 描述             |
| :----------------- | :---------------- |
| tf.add(x, y)      | 将x和y逐元素相加 |
| tf.subtract(x, y) | 将x和y逐元素相减 |
| tf.multiply(x, y) | 将x和y逐元素相乘 |
| tf.divide(x, y)   | 将x和y逐元素相除 |
| tf.floordiv(x, y)   | 将x和y逐元素相除取商，类似x//y |
| tf.math.mod(x, y) | 对x逐元素取模    |



```python
import tensorflow as tf

a = tf.constant([0, 1, 2, 3])
b = tf.constant([11, 12, 12, 15])
tf.add(a, b)
```




    <tf.Tensor: shape=(4,), dtype=int32, numpy=array([11, 13, 14, 18], dtype=int32)>




```python
tf.subtract(b, a)
```




    <tf.Tensor: shape=(4,), dtype=int32, numpy=array([11, 11, 10, 12], dtype=int32)>




```python
tf.multiply(a, b)
```




    <tf.Tensor: shape=(4,), dtype=int32, numpy=array([ 0, 12, 24, 45], dtype=int32)>




```python
tf.divide(a, b)
```




    <tf.Tensor: shape=(4,), dtype=float64, numpy=array([0.        , 0.08333333, 0.16666667, 0.2       ])>




```python
tf.math.mod(a, b)
```




    <tf.Tensor: shape=(4,), dtype=int32, numpy=array([0, 1, 2, 3], dtype=int32)>



**幂指对数运算**

| 算术操作       | 描述                  |
| :------------- | :-------------------- |
| tf.pow(x, y)   | 对x求y的幂次方        |
| tf.square(x)   | 将x逐元素求平方       |
| tf.sqrt(x)     | 将x逐元素求平方根     |
| tf.exp(x)      | 计算e的x次方          |
| tf.math.log(x) | 计算底数为e的自然对数 |



```python
a = tf.constant([0, 1, 2, 3])
tf.math.pow(a, 2)
```




    <tf.Tensor: shape=(4,), dtype=int32, numpy=array([0, 1, 4, 9], dtype=int32)>




```python
a = tf.constant([0, 1, 2, 3])
b = tf.constant([2, 3, 4, 2])
tf.pow(a, b)
```




    <tf.Tensor: shape=(4,), dtype=int32, numpy=array([ 0,  1, 16,  9], dtype=int32)>




```python
a = tf.constant([0, 1, 2, 3], dtype=tf.float32)
tf.pow(a, 0.5)
```




    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.       , 1.       , 1.4142135, 1.7320508], dtype=float32)>




```python
a = tf.constant([0, 1, 2, 3])
tf.math.square(a)
```




    <tf.Tensor: shape=(4,), dtype=int32, numpy=array([0, 1, 4, 9], dtype=int32)>




```python
a = tf.constant([0, 1, 2, 3], dtype=tf.float32)
tf.sqrt(a)
```




    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.       , 1.       , 1.4142135, 1.7320508], dtype=float32)>




```python
tf.exp(1.)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=2.7182817>




```python
a = tf.exp(3.)
tf.math.log(a)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=3.0>



tensorflow中只有以e的自然对数，如果要求以2为底的16的对数：


```python
a = tf.math.log(2.)
b = tf.math.log(16.)
b/a
```




    <tf.Tensor: shape=(), dtype=float32, numpy=4.0>




```python
a = tf.constant([[1., 9.], [16., 100]])
b = tf.constant([[2., 3.], [2., 10.]])
tf.math.log(a) / tf.math.log(b)
```




    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[0., 2.],
           [4., 2.]], dtype=float32)>



| 算术操作          | 描述                     |
| :---------------- | :----------------------- |
| tf.math.sign(x)        | 返回x的符号              |
| tf.math.abs(x)         | 将x逐元素求绝对值        |
| tf.math.negative(x)    | 将x逐元素相反数， y = -x |
| tf.math.reciprocal(x)  | 计算x的倒数        | 
| tf.math.logical_not(x) | 对x逐元素求逻辑非运算    |
| tf.math.ceil(x)        | 向上取整           |
| tf.math.floor(x)       | 向下取整                 |
| tf.math.rint(x)        | 取最接近的整数         |
| tf.math.round(x)       | 逐元素四舍五入最接近的整数   |
| tf.math.maximun(x, y)  | 返回两tensor中的最大值   |
| tf.math.minimun(x, y)  | 返回两tensor中的最小值   |


```python
tf.math.sign(a)
```




    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([1., 1., 1., 1.], dtype=float32)>




```python
a = tf.constant([0.1, 2.2, 4.5, 5.6])
tf.math.floor(a)
```




    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0., 2., 4., 5.], dtype=float32)>




```python
tf.math.ceil(a)
```




    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([1., 3., 5., 6.], dtype=float32)>




```python
tf.math.round(a)
```




    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0., 2., 4., 6.], dtype=float32)>




```python
tf.math.rint(a)
```




    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0., 2., 4., 6.], dtype=float32)>




```python
tf.math.negative(a)
```




    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([-0.1, -2.2, -4.5, -5.6], dtype=float32)>




```python
tf.math.reciprocal(a)
```




    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([10.        ,  0.45454544,  0.22222222,  0.17857143], dtype=float32)>




```python
tf.math.maximum(a, a)
```




    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.1, 2.2, 4.5, 5.6], dtype=float32)>



**三角函数与反三角函数运算**

| 函数       | 描述             |
| :--------- | :--------------- |
| tf.cos(x)  | 三角函数cos      |
| tf.sin(x)  | 三角函数sin      |
| tf.tan(x)  | 三角函数tan      |
| tf.acos(x) | 反三角函数arccos |
| tf.asin(x) | 反三角函数arcsin |
| tf.atan(x) | 反三角函数arctan |



```python
a = tf.constant([1., 2., 3.])
tf.math.sin(a)
```




    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.84147096, 0.9092974 , 0.14112   ], dtype=float32)>




```python
a = tf.constant([1., 2., 3.])
tf.sin(a)
```




    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.84147096, 0.9092974 , 0.14112   ], dtype=float32)>



**运算符重载**

| 运算符 | 构造方法                    |
| :----- | :-------------------------- |
| x + y  | tf.math.add(x, y)           |
| x - y  | tf.math.subtract(x, y)      |
| x * y  | tf.math.multiply(x, y)      |
| x / y  | tf.math.divide(x, y)        |
| x // y | tf.math.floordiv(x, y)      |
| x % y  | tf.math.mod(x, y)           |
| x ** y | tf.math.pow(x, y)           |
| -x     | tf.math.negative(x)         |
| abs(x) | tf.math.abs(x)              |
| x & y  | tf.math.logical_and(x, y)   |
| x \| y | tf.math.logical_or(x, y)    |
| x ^ y  | tf.math.logical_xor(x, y)   |
| ~ x    | tf.math.logical_not(x)      |
| x < y  | tf.math.less(x, y)          |
| x <= y | tf.math.less_equal(x, y)    |
| x > y  | tf.math.greater(x, y)       |
| x >= y | tf.math.greater_equal(x, y) |



```python
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
a * b
```




    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([ 4, 10, 18], dtype=int32)>




```python
a / b
```




    <tf.Tensor: shape=(3,), dtype=float64, numpy=array([0.25, 0.4 , 0.5 ])>




```python
a % b
```




    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>




```python
a // b
```




    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([0, 0, 0], dtype=int32)>




```python
a ** b
```




    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([  1,  32, 729], dtype=int32)>



**广播机制**

如果参与运算的两个张量维度不一致，也可以运算,但要求两个张量的最后一个维度必须相等。


```python
a = tf.constant([1, 2, 3])
b = tf.reshape(tf.range(12), shape=[4, 3])
a + b # a张量被广播到b张量的每一行上
```




    <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
    array([[ 1,  3,  5],
           [ 4,  6,  8],
           [ 7,  9, 11],
           [10, 12, 14]], dtype=int32)>




```python
a = tf.constant([1, 2, 3])
b = tf.reshape(tf.range(12), shape=[4, 3])
a * b
```




    <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
    array([[ 0,  2,  6],
           [ 3,  8, 15],
           [ 6, 14, 24],
           [ 9, 20, 33]], dtype=int32)>



当张量和一个数字运算时，则会将这个数字广播到张量的各个元素。


```python
a = tf.constant([1, 2, 3])
a * 5
```




    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([ 5, 10, 15], dtype=int32)>



**tensorflow张量与numpy数组类型自动转换**

当张量和numpy数组共同参与运算时：如果执行的是tensorflow操作，那么会自动将numpy数组转换为张量；如果执行的是numpy操作，那么会自动将张量转换为numpy数组;如果执行的是运算符操作，只要其中有一个是张量，就会自动将其他操作数都转换为张量，再进行运算。


```python
import numpy as np
import tensorflow as tf

a = np.arange(4).reshape(2, 2)
b = tf.reshape(tf.range(1, 5), shape=(2, 2))
tf.math.multiply(a, b) #执行tensorflow操作，则结果是张量
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[ 0,  2],
           [ 6, 12]], dtype=int32)>




```python
np.add(a, b) #执行numpy操作，则结果是numpy数组
```




    array([[1, 3],
           [5, 7]])




```python
a + b #操作数有一个张量，则结果也是张量
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[1, 3],
           [5, 7]], dtype=int32)>




```python
c = np.ones((2, 2)) 
a + c #操作数都是numpy数组，则结果也是numpy数组
```




    array([[1., 2.],
           [3., 4.]])



## 向量/矩阵运算

张量相乘是`tf.math.multiply()`或者`*`运算符

矩阵相乘是`tf.matmul()`或者`@`运算符


```python
a = tf.reshape(tf.range(6), shape=(2, 3))
b = tf.reshape(tf.range(12), shape=(3, 4))
tf.matmul(a, b)
```




    <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
    array([[20, 23, 26, 29],
           [56, 68, 80, 92]], dtype=int32)>




```python
a @ b
```




    <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
    array([[20, 23, 26, 29],
           [56, 68, 80, 92]], dtype=int32)>



由于a, b维度不同，a*b会出错

matmul()函数不在math模块下

**多维向量乘法**

如果两个张量维度不一样，也会用广播机制让最后相同的维度相乘，比如3维张量矩阵乘以2维张量


```python
a = tf.random.normal([2, 3, 4])
b = tf.random.normal([4, 2])
a @ b
```




    <tf.Tensor: shape=(2, 3, 2), dtype=float32, numpy=
    array([[[-1.2361232 , -1.2499229 ],
            [ 2.0451078 ,  1.2070524 ],
            [ 1.816917  ,  2.9748683 ]],
    
           [[-1.3890347 , -2.312685  ],
            [-1.0125731 , -4.8830366 ],
            [-0.22415736, -1.9630824 ]]], dtype=float32)>



3维张量矩阵乘以3维张量，4维张量矩阵乘以4维张量,也是后两维相乘


```python
a = tf.constant(np.arange(12).reshape(2, 3, 2))
b = tf.constant(np.arange(12).reshape(2, 2, 3))
a
```




    <tf.Tensor: shape=(2, 3, 2), dtype=int64, numpy=
    array([[[ 0,  1],
            [ 2,  3],
            [ 4,  5]],
    
           [[ 6,  7],
            [ 8,  9],
            [10, 11]]])>




```python
b
```




    <tf.Tensor: shape=(2, 2, 3), dtype=int64, numpy=
    array([[[ 0,  1,  2],
            [ 3,  4,  5]],
    
           [[ 6,  7,  8],
            [ 9, 10, 11]]])>




```python
a @ b
```




    <tf.Tensor: shape=(2, 3, 3), dtype=int64, numpy=
    array([[[  3,   4,   5],
            [  9,  14,  19],
            [ 15,  24,  33]],
    
           [[ 99, 112, 125],
            [129, 146, 163],
            [159, 180, 201]]])>



## 数据统计

求张量在某个维度或全局的统计值


| 函数                               | 描述     |
| ---------------------------------- | -------- |
| tf.reduce_sum(input_tensor, axis)  | 求和     |
| tf.reduce_mean(input_tensor, axis) | 求平均值 |
| tf.reduce_max(input_tensor, axis)  | 求最大值 |
| tf.reduce_min(input_tensor, axis)  | 求最小值 |


名字都有一个reduce，意思是降维


```python
a = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.reduce_sum(a)
```




    <tf.Tensor: shape=(), dtype=int32, numpy=21>




```python
a = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.reduce_sum(a, axis=0)
```




    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([5, 7, 9], dtype=int32)>




```python
a = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.reduce_sum(a, axis=1)
```




    <tf.Tensor: shape=(2,), dtype=int32, numpy=array([ 6, 15], dtype=int32)>




```python
a = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.reduce_mean(tf.cast(a, tf.float32))
```




    <tf.Tensor: shape=(), dtype=float32, numpy=3.5>




```python
a = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.reduce_mean(tf.cast(a, tf.float32), axis=0)
```




    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([2.5, 3.5, 4.5], dtype=float32)>




```python
a = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.reduce_mean(tf.cast(a, tf.float32), axis=1)
```




    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([2., 5.], dtype=float32)>




```python
a = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.reduce_max(a)
```




    <tf.Tensor: shape=(), dtype=int32, numpy=6>




```python
a = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.reduce_max(a, axis=0)
```




    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([4, 5, 6], dtype=int32)>




```python
a = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.reduce_max(a, axis=1)
```




    <tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 6], dtype=int32)>



求最值的索引

`tf.argmax()`

`tf.argmin()`


```python
a = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.argmax(a) #没有指定axis时，默认axis=0
```




    <tf.Tensor: shape=(3,), dtype=int64, numpy=array([1, 1, 1])>




```python
a = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.argmax(a, axis=0)
```




    <tf.Tensor: shape=(3,), dtype=int64, numpy=array([1, 1, 1])>




```python
a = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.argmax(a, axis=1)
```




    <tf.Tensor: shape=(2,), dtype=int64, numpy=array([2, 2])>


