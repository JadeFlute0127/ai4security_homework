# ai4security_homework ：CIFAR10图像分类

# 模型介绍
使用Pytorch框架，模型ResNet，使用交叉墒作为Loss函数，优化器选择SGD/Aam。
将Accuracy作为模型评估的标准，在CIFAR10数据集中进行训练。
```py
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
```

# 调参
### 优化器
1. SGD(随机梯度下降)  每次只选择一个样本的数据来进行更新梯度
2. Adam  泛化极好，比起其他优化器性能也更好，会对梯度、梯度的平方使用指数加权平均。

### BatchSize
batchsize指的是：单次训练所选取的样本数。
训练样本是从训练集中随机选取的，因此训练样本满足独立同分布。
适当增加BatchSize能够减少样本的方差，使得梯度可以更加准确。

```py
batchSize=16
Optimizer=Adam
```
![16](https://user-images.githubusercontent.com/35321989/137672066-8ce23236-e8f3-40bb-a7fe-993919ed549a.png)

```py
batchSize=32
Optimizer=Adam
```
![32](https://user-images.githubusercontent.com/35321989/137671681-66170698-2424-4816-b045-1559588da88e.png)

```py
batchSize=64
Optimizer=Adam
```

![64](https://user-images.githubusercontent.com/35321989/137671676-e9bb9baa-bb87-4be3-a9f3-70a5899a8541.png)

