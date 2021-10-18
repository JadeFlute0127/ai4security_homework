# ai4security_homework ：CIFAR10图像分类

# 模型介绍
使用Pytorch框架，模型选用6层CNN，使用交叉墒作为Loss函数，优化器选择SGD/Aam。
将Accuracy作为模型评估的标准，在CIFAR10数据集中进行训练。
```py
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
```

# 超参数调整
### 优化器
1. SGD(随机梯度下降)  每次只选择一个样本的数据来进行更新梯度
2. Adam  泛化极好，比起其他优化器性能也更好，会对梯度、梯度的平方使用指数加权平均。

### BatchSize
batchsize指的是：一次训练所选取的样本数。
训练样本是从训练集中随机选取的，因此训练样本满足独立同分布。
适当增大BatchSIze能够减少样本的方差，因此梯度可以更加准确。

```py
batchSize=16
Optimizer=Adam
```

```py
batchSize=32
Optimizer=Adam
```


```py
batchSize=64
Optimizer=Adam
```
