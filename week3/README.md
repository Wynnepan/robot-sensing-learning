# 第 3 周：数学最小集

本文件夹包含四个适合初学者的小程序，分别练习坐标变换、手写梯度下降、低通滤波，以及用 PyTorch 训练 MNIST 手写数字分类模型。程序使用 PyTorch tensor，并在代码中打印关键训练或计算信息。

## 文件说明

1. `coordinate_transform_2d.py`
   - 解决问题：把机器人局部坐标系中的二维点，批量变换到世界坐标系。
   - 输出图片：`outputs/coordinate_transform_2d.png`

2. `linear_regression_manual_gd.py`
   - 解决问题：不使用 `torch.optim`，手工用梯度下降拟合 `y = wx + b`。
   - 输出图片：`outputs/linear_regression_fit.png`、`outputs/linear_regression_loss.png`

3. `low_pass_filter.py`
   - 解决问题：给带高斯噪声的传感器信号做一阶低通滤波，并比较不同 `alpha` 的效果。
   - 输出图片：`outputs/low_pass_filter_comparison.png`

4. `train_mnist_pytorch.py`
   - 解决问题：下载 MNIST 数据集，训练一个小型 CNN 做 0 到 9 手写数字分类。
   - 默认数据目录：`data/mnist`
   - 默认模型输出：`outputs/mnist/mnist_cnn.pt`

## 依赖安装

推荐先创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

安装依赖：

```bash
pip install torch matplotlib
```

如果当前环境缺少 PyTorch，可以先尝试：

```bash
pip install torch
```

如果安装失败，建议打开 PyTorch 官网安装页，根据自己的系统、Python 版本、CPU/GPU 环境复制对应命令。初学阶段只跑这些小程序，CPU 版本就足够。

## 运行命令

```bash
python coordinate_transform_2d.py
python linear_regression_manual_gd.py
python low_pass_filter.py
python train_mnist_pytorch.py --epochs 1 --device cpu
```

MNIST 训练脚本常用参数：

```bash
python train_mnist_pytorch.py --epochs 3 --batch-size 128 --lr 0.001 --device cpu
```

如果本机有可用 CUDA，也可以运行：

```bash
python train_mnist_pytorch.py --epochs 3 --device cuda
```

## 核心公式

### 1. 二维坐标变换

旋转矩阵：

```text
R = [[cos(theta), -sin(theta)],
     [sin(theta),  cos(theta)]]
```

单点列向量形式：

```text
point_world = R @ point_robot + translation
```

批量行向量形式：

```text
points_world = points_robot @ R.T + translation
```

`points_robot` 的 shape 是 `[N, 2]`，每一行是一个点。因为点被按行保存，矩阵乘法时点在左边，所以要用 `R.T`。如果忘记转置，旋转方向和结果都会错。

### 2. 手写线性回归

预测：

```text
y_pred = w * x + b
```

误差：

```text
error = y_pred - y
```

均方误差 MSE：

```text
loss = mean((y_pred - y) ** 2)
```

手工梯度：

```text
grad_w = 2 * mean((y_pred - y) * x)
grad_b = 2 * mean(y_pred - y)
```

参数更新：

```text
w = w - learning_rate * grad_w
b = b - learning_rate * grad_b
```

### 3. 一阶低通滤波

递推公式：

```text
y[k] = y[k-1] + alpha * (x[k] - y[k-1])
```

`alpha` 越小，每一步接收的新观测越少，所以曲线更平滑；但真实信号变化时，输出追上去也更慢，因此延迟更明显。

由截止频率计算 `alpha`：

```text
tau = 1 / (2 * pi * fc)
alpha = dt / (tau + dt)
```

## 三个主题之间的联系

坐标变换、梯度下降、概率采样和低通滤波都在处理“用数学模型解释或修正数据”。

- 坐标变换：把同一个点从机器人坐标系表达成世界坐标系，本质是线性变换加平移。
- 梯度下降：通过 loss 衡量预测和数据的差距，再沿着让 loss 下降的方向调整参数。
- 概率采样：高斯噪声用随机采样模拟真实传感器和数据中的不确定性。
- 低通滤波：在噪声数据中保留慢变化，压制快速抖动，用递推公式不断更新估计值。

机器人学习中经常同时遇到这些内容：传感器有噪声，数据需要滤波；机器人坐标要变换到世界坐标；模型参数要通过优化方法从数据中学习。

## 常见错误

- 混淆 `@` 和 `*`：`@` 是矩阵乘法，`*` 是逐元素相乘。
- 批量点忘记 `R.T`：`[N, 2] @ [2, 2]` 使用的是行向量形式，要用 `R.T`。
- 旋转和平移顺序错误：通常先旋转局部点，再加世界平移。
- 梯度正负号错误：梯度下降是 `参数 = 参数 - learning_rate * 梯度`。
- 学习率过大：loss 可能震荡或发散。
- `backward()` 后忘记清空梯度：使用 autograd 训练时，梯度会累加，需要 `zero_grad()`；本项目训练主体没有用 `backward()`。
- 混淆 `epsilon` 与 `learning_rate`：`epsilon` 用于数值梯度检查的小扰动，`learning_rate` 用于更新参数。
- 混淆方差与标准差：`torch.randn` 生成标准正态噪声，乘上的系数是标准差。
- 混淆采样频率与信号频率：采样频率是每秒采多少点，信号频率是信号本身每秒变化多少周期。
- `alpha` 太小导致延迟明显：滤波更平滑，但跟踪阶跃变化会更慢。

## 完成检查表

- [ ] 能运行 `python coordinate_transform_2d.py`
- [ ] 能运行 `python linear_regression_manual_gd.py`
- [ ] 能运行 `python low_pass_filter.py`
- [ ] 能运行 `python train_mnist_pytorch.py --epochs 1 --device cpu`
- [ ] `outputs/coordinate_transform_2d.png` 存在且非空
- [ ] `outputs/linear_regression_fit.png` 存在且非空
- [ ] `outputs/linear_regression_loss.png` 存在且非空
- [ ] `outputs/low_pass_filter_comparison.png` 存在且非空
- [ ] `outputs/mnist/mnist_cnn.pt` 存在且非空
- [ ] 坐标变换中单点结果和批量结果一致
- [ ] 线性回归拟合参数接近 `w=2, b=1`
- [ ] 数值梯度和手工梯度接近
- [ ] 低通滤波输出没有 NaN

## 自测题

1. `points_robot` 的 shape 是 `[N, 2]` 时，`N` 表示什么？
2. 为什么批量坐标变换要写 `points_robot @ R.T`？
3. 如果先平移再旋转，结果和先旋转再平移一样吗？为什么？
4. `@` 和 `*` 在 PyTorch 中分别表示什么？
5. MSE 为什么要对误差平方求平均？
6. 梯度下降更新参数时，为什么要减去梯度，而不是加上梯度？
7. 数值梯度检查里的 `epsilon` 和训练里的 `learning_rate` 有什么区别？
8. `torch.randn_like(x) * 0.35` 中的 `0.35` 是方差还是标准差？
9. 采样频率 `fs=50 Hz` 表示什么？它和正弦信号频率有什么区别？
10. 低通滤波中 `alpha=0.02` 和 `alpha=0.50` 哪个更平滑？哪个延迟更大？
