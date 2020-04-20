# 学习中的问题

### Q：在假设函数为多元线性方程和非线性方程时手动测试Batch gradient descent 算法时不能收敛，不知道是训练样本的问题还是学习率的问题？
    
    具体参考Chapter1.ipynb
    
    PS:尝试了0.001~0.1不同的学习率均不能收敛。
___    
    
### Q：不同平台Python运行速度不同？
    
    平时常用的Emacs编辑器的shell模块以及python插件来运行，每秒钟迭代几百次，由于问题1不能收敛，运行了6个小时才迭代了10w次左右；然后我用pycharm和系统terminal都跑了一次，每秒钟迭代几万次。

### A：

进行了以下测试：

以同样的代码在不同平台进行测试，设置迭代次数为2w次，运行10次，计算各平台测试的10次的平均时间。

```python
# 测试代码
import time

def run():
    x = [[1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2],
         [3, 3, 3, 3, 3],
         [4, 4, 4, 4, 4],
         [5, 5, 5, 5, 5]]
    y = [1.8, 3.6, 5.4, 7.2, 9]
    theta = [0, 0, 0, 0, 0, 0]
    m = len(x)  # number of training examples
    alpha = 0.001  # learning rate
    threshold = 0.001  # error control
    cost_dict = {}  # cost function result
    iter_num = 0  # iteration times

    def cost_j(theta):
        tmp1 = 0
        tmp2 = 0
        for i in range(m):
            tmp1 += theta[0]
            for j in range(m):
                tmp1 += theta[j+1] * x[i][j]
            tmp2 += (tmp1 - y[i])**2
        return tmp2/(2*m)

    def gradient_descent(theta):
        tmp = [0, 0, 0, 0, 0, 0]
        tmp1 = 0
        for i in range(m):
            tmp1 = theta[0]
            for j in range(m):
                tmp1 += theta[j+1] * x[i][j]
            tmp1 -= y[i]
            tmp[0] += tmp1  # 对应 theta[0]
            for j in range(m):
                # 对应 theta[1] ~ theta[5]
                tmp[j+1] += tmp1 * x[i][j]
        theta[0] -= alpha * tmp[0] / m
        for j in range(m):
            theta[j+1] -= alpha * tmp[j+1] / m
        return theta

    cost = cost_j(theta)
    print("initial cost = {}".format(cost))

    while cost > threshold:
        theta = gradient_descent(theta)
        iter_num += 1
        cost = cost_j(theta)
        cost_dict[iter_num] = cost
        print("iteration {}, theta = {}".format(iter_num, theta))
        if iter_num > 20000: # 迭代次数设置为20000次
            break

if __name__ == "__main__":
    start = time.time()
    for i in range(10):
        run()
    end = time.time()

    print("Average time: {}".format((end-start)/10))
```
不同平台测试结果为：

| Platform | Average time used (s) |
| :--- | :--- |
| VSCode + Python插件 | 2.1939486980438234 |
| Emacs + elpy插件| 54.190464806556705 |
| Emacs + eshell | 73.82958469390869 |
| 系统 Terminal + python | 1.3879480123519898|
| Pycharm | 0.8409967660903931 |
| Pycharm terminal+python | 2.26530978679657 |

结果很惊讶！让一直用Emacs的我很无语！

---

