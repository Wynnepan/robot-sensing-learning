# 学习进度与交接笔记

> 更新日期：2026-08-13  
> 学习方向：机器人 / PyTorch / 机器学习基础  
> 用途：交接给下一位辅导者，同时作为个人复习笔记

## 1. 当前进度总览

目前已经推进到第 1 个月学习计划的后半段，已经完成或讲解过：

1. 第 1 周 Python、NumPy、matplotlib 和 Git 基础，并完成代码作业；
2. 第 2 周 PyTorch 入门，并完成线性回归与 MLP 代码作业；
3. 第 3 周“数学最小集”的完整知识讲解；
4. “机器学习最小集”的监督学习、数据集划分、过拟合、分类、神经网络层和交叉熵等核心概念。

当前即将进入的实践任务是：

1. 用 PyTorch 完整训练 MNIST；
2. 写笔记《我理解的 PyTorch 训练流程》；
3. 后续按要求选看 CS285 的 Introduction、Imitation Learning、RL Basics，只理解概念，不深挖推导。

第 1～2 周代码作业位于 [robot-sensing-learning GitHub 仓库](https://github.com/Wynnepan/robot-sensing-learning/tree/main)。本次交接已实际克隆并核对该仓库；第 3 周三个综合程序尚未在当前工作区落盘，因此应把第 3 周视为“知识已讲完，综合编码作业尚未完成”。

---

## 2. 学习者情况与讲解偏好

学习者目前处于入门阶段，但会认真追问公式和代码中每个中间环节。

适合的讲解方法：

- 从小数字、小矩阵、小数据集开始，再推广到 MNIST；
- 解释每个 tensor 的 shape，以及每个维度代表什么；
- 代码逐句解释，不要只给完整程序；
- 新术语不要省略，例如 batch、iteration、epoch 要一起解释；
- 数学公式要同时翻译成普通语言；
- 清楚区分相似概念，例如参数与超参数、epsilon 与 learning rate、logits 与概率；
- 每次只推进一个主要知识点，确认理解后再继续；
- 解释“为什么要这样做”，不能只解释“代码做了什么”。

已发现容易卡住的地方：

- 坐标变换顺序；
- 批量点为什么使用 `R.T`；
- 梯度下降为什么减去梯度；
- 数值梯度到底有什么用；
- batch、iteration、epoch 的层级关系；
- tensor 的 `dim`；
- 神经网络中各层的含义与 `nn.Linear()` 两个数字的来源。

---

# 第一部分：第 1 个月原始学习计划

## 3. 月度目标

主题：Python、PyTorch 和数学最小基础。

原始目标：

- 能自己写 Python 小程序；
- 能使用 NumPy、matplotlib、PyTorch；
- 能理解神经网络训练的基本流程；
- 能用 Python 生成、滤波和可视化传感器信号。

三周学习路径：

```text
第1周：Python + NumPy + 信号处理小程序
    ↓
第2周：PyTorch tensor + 自动求导 + 训练循环
    ↓
第3周：坐标变换 + 梯度数学 + 概率采样与滤波
    ↓
当前：机器学习分类最小集 + 即将训练 MNIST
```

---

# 第二部分：第 1 周 Python + NumPy

## 4. 第 1 周计划与完成状态

学习内容：

- Python 变量、函数、`list`、`dict`、`class`；
- NumPy array、shape、矩阵运算；
- matplotlib 画图；
- Git 基础。

原始任务及状态：

| 任务 | 状态 | 仓库文件 |
|---|---|---|
| 画模拟传感器信号曲线 | 已完成 | `week1/sensor_signal.py` |
| 给模拟信号加噪声 | 已完成 | `week1/sensor_signal.py` |
| 写 moving average filter | 已完成 | `week1/moving_average.py` |
| 用 NumPy 写 softmax | 已完成 | `week1/softmax.py` |
| 写一个 2D 旋转矩阵 | 已完成 | `week1/rotation_matrix.py` |
| 建 GitHub repo | 已完成 | `Wynnepan/robot-sensing-learning` |

## 5. Python 基础复习

### 5.1 变量

变量用名称保存一个值：

```python
frequency = 1.0
window_size = 20
name = "sensor"
```

Python 根据赋给变量的值确定类型。变量可以保存数字、字符串、列表、字典、NumPy 数组、tensor 或模型等对象。

### 5.2 函数

函数把一段可复用逻辑封装起来：

```python
def moving_average(signal, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode="same")
```

需要理解：

- `def` 定义函数；
- 括号内是输入参数；
- `return` 返回结果；
- docstring 和注释说明函数用途；
- 函数可以对输入进行检查，并在输入非法时 `raise ValueError`。

### 5.3 list、dict 和 class

`list` 是有顺序、可修改的数据集合：

```python
loss_history = [0.8, 0.5, 0.3]
```

`dict` 使用键值对组织数据：

```python
history = {
    "train_loss": [],
    "validation_loss": [],
}
```

`class` 用于把数据和相关行为组织为对象：

```python
class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.output = None

    def update(self, measurement):
        ...
```

- `__init__`：创建对象时设置初始状态；
- `self`：当前对象自身；
- 属性保存状态；
- 方法描述对象能执行的操作。

第 1 周仓库作业主要使用变量和函数；`list`、`dict`、`class` 属于已列入学习范围的 Python 基础，后续会在训练历史、配置和在线滤波器中继续巩固。

## 6. NumPy array、shape 和运算

NumPy array 是适合数值计算的多维数组：

```python
point = np.array([1.0, 0.0])
matrix = np.array([
    [0.0, -1.0],
    [1.0,  0.0],
])
```

`shape` 表示每个维度的大小：

```text
point.shape  = (2,)
matrix.shape = (2, 2)
```

常见创建函数：

```python
np.linspace(start, stop, number)  # 均匀生成指定数量的点
np.ones(n)                        # 全1数组
np.zeros(n)                       # 全0数组
np.asarray(x, dtype=float)        # 转为数组
```

常见运算：

```python
a + b          # 逐元素加法
a * b          # 逐元素乘法
matrix @ point # 矩阵乘法
np.mean(x)     # 平均值
np.sum(x)      # 求和
np.max(x)      # 最大值
np.argmax(x)   # 最大值的位置
```

## 7. 第 1 周实际代码复盘

### 7.1 模拟传感器信号与高斯噪声

仓库文件：`week1/sensor_signal.py`。

时间轴：

```python
t = np.linspace(0, 10, 1000)
```

干净的 1 Hz 正弦信号：

```python
clean_signal = np.sin(
    2 * np.pi * frequency * t
)
```

固定随机种子并生成高斯噪声：

```python
rng = np.random.default_rng(seed=42)
noise = rng.normal(
    loc=0.0,
    scale=0.3,
    size=t.shape,
)
```

- `loc=0.0`：噪声均值；
- `scale=0.3`：噪声标准差；
- `size=t.shape`：噪声与时间轴 shape 相同。

测量信号：

```python
noisy_signal = clean_signal + noise
```

该作业已经把 NumPy、随机数、高斯噪声和 matplotlib 曲线连接起来。

### 7.2 matplotlib 画图

仓库代码已经用过：

```python
plt.figure(figsize=(12, 6))
plt.plot(t, clean_signal, label="Clean signal")
plt.plot(t, noisy_signal, label="Noisy signal")
plt.xlabel("Time (s)")
plt.ylabel("Sensor Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("week1/sensor_signal.png", dpi=150)
```

需要掌握：

- `figure` 创建画布；
- `plot` 画曲线；
- `scatter` 画散点；
- `label` 配合 `legend` 显示图例；
- `alpha` 控制透明度；
- `savefig` 保存图片；
- `close` 关闭图像，避免脚本批量运行时占用资源。

### 7.3 Moving Average Filter

仓库文件：`week1/moving_average.py`。

长度为 `window_size` 的平均滤波核：

```python
kernel = np.ones(window_size) / window_size
```

例如窗口为 4：

\[
kernel=[0.25,0.25,0.25,0.25]
\]

卷积滤波：

```python
filtered_signal = np.convolve(
    signal,
    kernel,
    mode="same",
)
```

直觉：窗口移动到每个位置，对附近数据求平均，从而削弱快速随机波动。

- 窗口小：响应快，平滑较弱；
- 窗口大：更平滑，但延迟和边界影响更明显；
- `mode="same"`：输出长度与输入相同。

该程序还检查了 `window_size` 必须大于 0 且不能超过信号长度。

### 7.4 NumPy Softmax

仓库文件：`week1/softmax.py`。

公式：

\[
p_i=\frac{e^{x_i}}{\sum_j e^{x_j}}
\]

数值稳定写法：

```python
shifted_x = x - np.max(x)
exp_x = np.exp(shifted_x)
probabilities = exp_x / np.sum(exp_x)
```

减去最大值不会改变 Softmax 结果，却能降低 `exp` 数值溢出的风险。

已验证内容：

- 输出全部为正；
- 所有概率之和为 1；
- `np.argmax(probabilities)` 得到最大概率类别位置。

这一作业与后来学习的分类、logits、Softmax 和交叉熵直接衔接。

### 7.5 NumPy 二维旋转矩阵

仓库文件：`week1/rotation_matrix.py`。

代码将角度转为弧度：

```python
angle_radians = np.deg2rad(angle_degrees)
```

构造逆时针旋转矩阵：

\[
R=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}
\]

旋转点：

```python
rotated_point = rotation @ point
```

程序测试了 `0、90、180、270、360、-90` 度，并使用 `np.round` 处理接近 0 的浮点误差。这项作业为第 3 周坐标变换奠定了基础。

## 8. Git 基础与仓库状态

已建立公开仓库：[Wynnepan/robot-sensing-learning](https://github.com/Wynnepan/robot-sensing-learning/tree/main)。

已核对的仓库结构：

```text
robot-sensing-learning/
├── .gitignore
├── README.md
├── requirements.txt
├── week1/
│   ├── sensor_signal.py
│   ├── moving_average.py
│   ├── softmax.py
│   └── rotation_matrix.py
└── week2/
    ├── tensor_and_linear.py
    └── mlp_noisy_function.py
```

Git 基础需要继续保持：

```text
git status   查看工作区状态
git add      暂存改动
git commit   保存一个版本
git push     推送到 GitHub
git log      查看提交历史
```

交接提醒：仓库当前 `README.md` 没有形成完整说明，`requirements.txt` 列出了 NumPy、SciPy、matplotlib、pandas、SymPy、Jupyter 和 ipykernel，但没有列出 `torch`。后续整理仓库时应补充 README、PyTorch 依赖和运行方式。

---

# 第三部分：第 2 周 PyTorch 入门

## 9. 第 2 周计划与完成状态

学习内容：

- tensor；
- autograd；
- loss；
- optimizer；
- training loop。

任务状态：

| 任务 | 状态 | 仓库文件 |
|---|---|---|
| PyTorch 拟合 `y = 3x + 2` | 已完成 | `week2/tensor_and_linear.py` |
| MLP 拟合带噪声的非线性函数 | 已完成 | `week2/mlp_noisy_function.py` |

## 10. Tensor 基础

Tensor 是 PyTorch 的多维数组，是模型、数据和梯度的基本容器。

```python
scalar = torch.tensor(5.0)  # 0维
vector = torch.tensor([5.0])  # 1维
matrix = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
])  # 2维
```

需要区分：

- `.shape`：各维度大小；
- `.ndim`：有几个维度；
- `.reshape(...)`：重塑形状，但元素总数不变；
- `.squeeze()`：删除大小为 1 的维度；
- `.numpy()`：CPU tensor 转为 NumPy array，便于 matplotlib 绘图。

第 2 周线性拟合作业使用：

```python
x = torch.linspace(-5, 5, 100).reshape(-1, 1)
```

`torch.linspace` 生成 100 个数，原 shape 为 `[100]`；`reshape(-1, 1)` 转为 100 行 1 列，即 `[100, 1]`。`-1` 表示让 PyTorch 自动推断该维度。

## 11. PyTorch 拟合 `y = 3x + 2`

### 11.1 数据与模型

```python
x = torch.linspace(-5, 5, 100).reshape(-1, 1)
y = 3 * x + 2
model = nn.Linear(1, 1)
```

`nn.Linear(1, 1)` 表示每个样本有 1 个输入特征并输出 1 个数，模型为：

\[
\hat y=wx+b
\]

模型最初随机初始化 `w,b`，训练目标是学到：

\[
w\approx3,\qquad b\approx2
\]

### 11.2 Loss 与 optimizer

```python
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
)
```

- `MSELoss`：衡量预测与真实值的平均平方差；
- `model.parameters()`：提供模型中需要训练的权重和偏置；
- `SGD`：随机梯度下降优化器；
- `lr`：学习率。

### 11.3 Training loop

仓库代码的训练顺序：

```python
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(x)
    loss = loss_function(predictions, y)
    loss.backward()
    optimizer.step()
```

含义：

1. 清空上一次梯度；
2. 前向传播得到预测；
3. 计算损失；
4. autograd 反向计算梯度；
5. optimizer 更新 `w,b`。

由于该作业每次把全部 100 个点交给模型，它属于 full-batch gradient descent。每个 epoch 只有一个 batch，因此普通情况下也是一个 iteration。

### 11.4 推理与绘图

```python
with torch.no_grad():
    final_predictions = model(x)
```

推理阶段不需要记录梯度，使用 `torch.no_grad()` 可节省计算和内存。然后用 `.squeeze().numpy()` 转为 matplotlib 可绘制的数据。

## 12. MLP 拟合带噪声的 `sin(x)`

### 12.1 数据

仓库任务拟合：

\[
y=\sin(x)+噪声
\]

代码：

```python
x = torch.linspace(-5, 5, 100).reshape(-1, 1)
y_clean = torch.sin(x)
noise = 0.1 * torch.randn_like(y_clean)
y_noisy = y_clean + noise
```

`torch.randn_like(y_clean)` 生成与 `y_clean` shape 相同的标准高斯噪声。

### 12.2 模型结构

```python
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1),
)
```

shape 流程：

```text
[100, 1]
→ Linear(1, 32)
[100, 32]
→ Tanh
[100, 32]
→ Linear(32, 32)
[100, 32]
→ Tanh
[100, 32]
→ Linear(32, 1)
[100, 1]
```

- 两个 32 是隐藏层宽度；
- `Tanh` 提供非线性，使网络能够拟合弯曲的 `sin(x)`；
- 最后一层输出 1 个连续值，所以这是回归任务。

### 12.3 Adam 与训练

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
)
```

Adam 也是优化器，会根据梯度更新参数，并为每个参数维护自适应更新信息。此阶段不要求推导 Adam，只需要理解 optimizer 的职责仍是更新参数。

训练完成后：

```python
model.eval()
with torch.no_grad():
    predictions = model(x)
```

- `model.eval()`：进入评估模式；
- `torch.no_grad()`：关闭梯度记录。

程序绘制了带噪散点、真实 `sin(x)` 和 MLP 预测曲线。

## 13. 第 2 周与后续知识的连接

第 2 周已经实际走过完整训练流程：

```text
构造数据
→ 定义模型
→ 定义 loss
→ 定义 optimizer
→ 前向传播
→ backward
→ optimizer.step
→ eval/no_grad
→ 可视化结果
```

第 3 周进一步解释了这些代码背后的数学：导数、梯度、链式法则、MSE 手工梯度、数值差分和 autograd。当前机器学习最小集又把相同流程从回归扩展到分类和 MNIST。

---

# 第四部分：第 3 周“数学最小集”

## 14. 原始任务与调整后的学习结构

第 3 周原始内容是向量、矩阵、坐标变换、导数、梯度、链式法则、期望、方差、高斯分布、采样和离散时间信号。原始任务是写二维坐标变换、用梯度下降拟合数据，以及写一阶低通滤波器比较滤波前后的传感器曲线。

后来为了让知识顺序更清晰，调整为三个模块：

### 模块一：向量、矩阵与坐标变换

学习标量、向量、矩阵、shape、转置、向量运算、点积、矩阵乘法、二维旋转和平移、坐标系、变换顺序和齐次坐标。任务是把机器人局部坐标系中的点转换到世界坐标系，并画出变换前后的点。

### 模块二：导数、梯度与梯度下降

学习函数、导数和斜率、偏导数、梯度、学习率、链式法则、数值梯度，以及 PyTorch autograd 与手工梯度的关系。任务是不使用 PyTorch optimizer，手工梯度下降拟合 `y = wx + b`，并用数值差分检查手算梯度。

### 模块三：概率、采样与滤波

学习均值与期望、方差和标准差、高斯分布、随机噪声、连续与离散时间、采样周期和采样频率、混叠以及一阶低通滤波器。任务是生成带高斯噪声的传感器信号，实现滤波器，比较真实、带噪和滤波信号，并改变参数观察平滑程度和延迟。

## 15. 向量、矩阵与坐标变换

### 15.1 基础对象

- 标量：一个数；
- 向量：一组有顺序的数；
- 矩阵：按行列排列的数，也可以理解为对向量执行变换的工具；
- `shape`：说明 tensor 每个维度有多大；
- 转置：交换矩阵的行与列；
- `*`：逐元素乘法；
- `@`：矩阵乘法。

点积：

\[
\mathbf a\cdot\mathbf b
=a_xb_x+a_yb_y
\]

点积可以反映两个向量的方向关系：大于 0 大致同向，等于 0 垂直，小于 0 大致反向。

### 15.2 二维旋转

二维逆时针旋转矩阵：

\[
R(\theta)=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}
\]

单个列向量点的旋转：

\[
\mathbf p'=R\mathbf p
\]

PyTorch：

```python
point_rotated = R @ point
```

### 15.3 旋转和平移

机器人局部坐标中的点转换到世界坐标：

\[
\boxed{\mathbf p_w=R_{wr}\mathbf p_r+\mathbf t_{wr}}
\]

含义：

1. 先用 `R` 把局部点旋转到世界坐标的方向；
2. 再用 `t` 把点移动到机器人在世界中的位置。

变换顺序很重要：

\[
R\mathbf p+\mathbf t
\ne
R(\mathbf p+\mathbf t)
\]

因为：

\[
R(\mathbf p+\mathbf t)=R\mathbf p+R\mathbf t
\]

右边把平移向量也旋转了，物理含义已经不同。

### 15.4 批量变换多个点

多个点通常按行保存：

```python
points.shape == [N, 2]
```

每一行是一个点 `[x, y]`。批量变换公式：

\[
\boxed{P_w=P_rR^T+\mathbf t}
\]

PyTorch：

```python
points_world = points_robot @ R.T + translation
```

为什么是 `R.T`：数学公式通常把单点写成列向量 `R @ point`；代码把多个点按行保存，相当于把列向量转成了行向量，因此等价写法变成 `point_row @ R.T`。

如果多个点按列保存为 `[2, N]`，则可以写：

```python
points_world = R @ points
```

### 15.5 齐次坐标

二维点后补一个 1：

\[
[x,y]\rightarrow[x,y,1]
\]

把旋转和平移放入一个矩阵：

\[
T=
\begin{bmatrix}
R&\mathbf t\\
0&1
\end{bmatrix}
\]

于是旋转和平移可以统一为一次矩阵乘法：

\[
\tilde{\mathbf p}_w=T\tilde{\mathbf p}_r
\]

齐次坐标的主要价值是方便组合多个坐标系变换。

---

## 16. 导数、梯度、链式法则与梯度下降

### 16.1 导数

导数表示函数在某一点的瞬时斜率：

\[
f'(x)=\frac{df}{dx}
\]

例如：

\[
f(x)=x^2,\qquad f'(x)=2x
\]

导数的符号表示局部方向：

- 导数为正：向右走，函数上升；
- 导数为负：向右走，函数下降；
- 导数为零：可能位于平坦点或极值点。

### 16.2 梯度下降为什么减去梯度

更新公式：

\[
\boxed{w\leftarrow w-\eta\frac{dL}{dw}}
\]

其中 `η` 是学习率。

梯度指向损失上升最快的方向；为了减小损失，要走向相反方向，所以公式中使用减号。

学习率控制每次真正更新参数的步长：

- 太小：训练慢；
- 合适：稳定下降；
- 太大：可能震荡或发散。

### 16.3 偏导数和梯度

当损失依赖多个参数：

\[
L=L(w,b)
\]

分别考察每个参数方向：

\[
\frac{\partial L}{\partial w},
\qquad
\frac{\partial L}{\partial b}
\]

把所有偏导数组合起来叫梯度：

\[
\nabla L=
\begin{bmatrix}
\frac{\partial L}{\partial w}\\
\frac{\partial L}{\partial b}
\end{bmatrix}
\]

### 16.4 链式法则

如果计算经过多层：

```text
w → z → L
```

则：

\[
\boxed{
\frac{dL}{dw}
=
\frac{dL}{dz}
\frac{dz}{dw}
}
\]

直线模型：

\[
\hat y=wx+b
\]

单个样本平方误差：

\[
L=(\hat y-y)^2
\]

计算路径：

```text
w,b → y_pred → error → loss
```

链式法则得到：

\[
\boxed{
\frac{\partial L}{\partial w}
=2(\hat y-y)x
}
\]

\[
\boxed{
\frac{\partial L}{\partial b}
=2(\hat y-y)
}
\]

### 16.5 一组数据的线性拟合

均方误差：

\[
\boxed{
\operatorname{MSE}
=\frac1N\sum_{i=1}^{N}(\hat y_i-y_i)^2
}
\]

手工梯度：

```python
error = y_pred - y
grad_w = 2 * torch.mean(error * x)
grad_b = 2 * torch.mean(error)
```

参数更新：

```python
w = w - learning_rate * grad_w
b = b - learning_rate * grad_b
```

### 16.6 数值梯度

中心差分：

\[
\boxed{
\frac{\partial L}{\partial w}
\approx
\frac{L(w+\epsilon)-L(w-\epsilon)}{2\epsilon}
}
\]

`epsilon` 是为了测量坡度而临时移动的很小距离，不是学习率。

- `epsilon`：用于探测和检查梯度；
- `learning_rate`：用于真正更新参数。

数值梯度计算慢，主要用来验证手工梯度是否正确。

### 16.7 PyTorch autograd

关键步骤：

```python
w = torch.tensor(0.0, requires_grad=True)
loss.backward()
print(w.grad)
```

- `requires_grad=True`：要求 PyTorch 跟踪相关计算；
- `loss.backward()`：沿计算图反向应用链式法则；
- `w.grad`：保存损失对 `w` 的梯度；
- `backward()` 只计算梯度，不自动更新参数；
- PyTorch 默认累积梯度，因此训练循环中要清空旧梯度。

手工更新参数时：

```python
with torch.no_grad():
    w -= learning_rate * w.grad

w.grad.zero_()
```

---

## 17. 概率、采样与滤波

### 17.1 均值、期望、方差和标准差

样本均值：

\[
\bar x=\frac1N\sum_{i=1}^{N}x_i
\]

期望是随机变量无限次重复实验的理论长期平均；现实中通常用样本均值估计期望：

\[
E[X]\approx\bar x
\]

方差：

\[
\operatorname{Var}(X)
=E[(X-E[X])^2]
\]

标准差：

\[
\sigma=\sqrt{\operatorname{Var}(X)}
\]

- 均值：数据中心在哪里；
- 方差：数据分散程度；
- 标准差：以原数据单位描述典型波动大小。

PyTorch 中为了和除以 `N` 的公式一致，可以明确写：

```python
torch.var(x, correction=0)
torch.std(x, correction=0)
```

### 17.2 高斯分布与噪声

高斯分布：

\[
X\sim\mathcal N(\mu,\sigma^2)
\]

- `μ`：分布中心；
- `σ`：标准差，决定波动大小；
- `σ²`：方差。

生成均值为 `mean`、标准差为 `std` 的高斯样本：

```python
samples = mean + std * torch.randn(number)
```

传感器模型：

\[
\boxed{测量值=真实值+噪声}
\]

零均值随机噪声经过多次测量可能正负抵消；固定 bias 不会被普通低通滤波自动消除。

### 17.3 连续与离散时间、采样

连续信号：

\[
x(t)
\]

离散采样：

\[
x[k]=x(kT_s)
\]

采样周期与采样频率：

\[
\boxed{f_s=\frac1{T_s}},
\qquad
\boxed{T_s=\frac1{f_s}}
\]

需要区分：

- 信号频率：信号每秒振动多少次；
- 采样频率：计算机每秒读取多少次。

基本奈奎斯特条件：

\[
f_s>2f_{\max}
\]

采样过慢会产生混叠，高频信号可能伪装成低频信号。混叠发生后，后续数字滤波通常不能可靠恢复已经丢失的信息。

### 17.4 一阶低通滤波器

公式：

\[
\boxed{
y[k]=y[k-1]+\alpha(x[k]-y[k-1])
}
\]

等价形式：

\[
y[k]=\alpha x[k]+(1-\alpha)y[k-1]
\]

含义：当前输出由一部分当前测量值和一部分历史输出组成。

- `alpha` 小：更平滑、延迟更大；
- `alpha` 大：响应更快、保留更多噪声；
- `alpha = 1`：输出等于输入，没有滤波效果。

实现核心：

```python
filtered[0] = signal[0]

for k in range(1, len(signal)):
    filtered[k] = (
        filtered[k - 1]
        + alpha * (signal[k] - filtered[k - 1])
    )
```

使用截止频率计算 `alpha`：

\[
\tau=\frac1{2\pi f_c}
\]

\[
\alpha=\frac{T_s}{\tau+T_s}
\]

---

## 18. 第 3 周实践任务状态

三个目标程序：

1. `coordinate_transform_2d.py`：批量转换机器人局部点并绘图；
2. `linear_regression_manual_gd.py`：不用 optimizer，手工梯度下降拟合带噪直线，并进行数值梯度检查；
3. `low_pass_filter.py`：生成带高斯噪声的传感器信号，比较多个 `alpha` 的平滑程度、延迟和 MSE。

状态：核心知识、公式、代码片段和完整实现要求已经讲解；当前工作区未发现这些程序文件，因此仍需实际创建、运行和验证。

---

# 第五部分：机器学习最小集

## 19. 监督学习

监督学习使用带答案的数据：

\[
D=\{(x_i,y_i)\}_{i=1}^{N}
\]

- `x`：输入；
- `y`：正确标签；
- `y_pred`：模型预测；
- 模型通过最小化损失学习参数。

两类常见任务：

- 回归：预测连续数值，常使用 MSE；
- 分类：预测离散类别，MNIST 是 10 分类问题。

---

## 20. Train / Validation / Test

### 20.1 三类数据集

- 训练集：学习模型权重和偏置；
- 验证集：选择学习率、batch size、层数、隐藏层宽度、epoch 数、停止时机等超参数；
- 测试集：所有方案确定之后，进行最终泛化评价。

模型参数由训练自动学习；超参数通常由开发者先设置，再根据验证集表现选择。

不能反复根据测试集结果调整模型，否则测试集也间接参与了模型选择，会造成数据泄漏。

### 20.2 随机划分

```python
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=generator
)
```

要点：

- `len(dataset)`：数据总数；
- `train_size`：训练子集大小；
- 测试大小使用减法，保证所有数据都被使用；
- `manual_seed(42)`：让随机划分能够复现；
- `random_split` 返回使用不同索引访问原数据的子集，通常不会复制全部数据。

MNIST 已经提供官方训练集和测试集。实际可从官方训练集中再划出验证集，并保留官方测试集到最后。

---

## 21. Dataset、DataLoader、Batch、Iteration、Epoch

### 21.1 Dataset 与 DataLoader

- `Dataset`：保存数据，能够通过索引取出一条样本；
- `DataLoader`：负责打乱数据、分 batch，并逐批送给模型。

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
```

`shuffle=True` 通常用于训练集，使每个 epoch 的样本顺序重新打乱。

### 21.2 四个训练层级

- Sample：一条样本；
- Batch：一次送给模型的一组样本；
- Iteration / Step：处理一个 batch，通常完成一次前向传播、反向传播和参数更新；
- Epoch：完整看完训练集一遍。

普通训练中可近似理解为：

\[
1\ batch=1\ iteration=1\ optimizer.step()
\]

每个 epoch 的 iteration 数：

\[
\boxed{
\left\lceil
\frac{训练样本数}{batch\ size}
\right\rceil
}
\]

MNIST 训练集有 60,000 张图片，`batch_size=64`：

\[
\left\lceil\frac{60000}{64}\right\rceil=938
\]

其中 937 个完整 batch 使用 59,968 张图片，最后一个 batch 有 32 张。因此：

```text
1 epoch = 938 batches = 938 iterations
```

如果训练 5 个 epoch，普通情况下总计约 4,690 次 iteration 和参数更新。

---

## 22. 过拟合

过拟合：训练集表现很好，但验证集或新数据表现较差。

典型现象：

```text
训练 loss 继续下降
验证 loss 先下降，后来上升
```

三种状态：

| 状态 | 训练表现 | 验证表现 |
|---|---|---|
| 欠拟合 | 差 | 差 |
| 合适 | 好 | 好，且差距较小 |
| 过拟合 | 很好 | 明显较差 |

常见缓解方法：

- 增加数据；
- 数据增强；
- 减小模型复杂度；
- 提前停止；
- L2 正则化 / `weight_decay`；
- Dropout。

训练停止时机应主要参考验证集，而不能只看训练 loss。

---

## 23. 分类、logits、Softmax 与准确率

### 23.1 分类输出

MNIST 有 10 个类别。模型对每张图片输出 10 个原始分数，称为 logits：

```text
logits.shape = [batch_size, 10]
```

logits：

- 可以为正或负；
- 不要求总和为 1；
- 不是概率；
- 越大表示模型越倾向对应类别。

预测类别：

```python
predictions = logits.argmax(dim=1)
```

### 23.2 `dim`

`dim` 表示沿哪个维度进行运算。

若：

```text
logits.shape = [64, 10]
```

则：

- `dim=0`：沿 64 个样本的方向；
- `dim=1`：沿 10 个类别的方向。

分类预测需要对每张图片的 10 个类别分数取最大值，因此使用 `dim=1`。

### 23.3 Softmax

Softmax 把 logits 转为总和为 1 的类别概率，并保持最大值所在类别不变。

```python
probabilities = torch.softmax(logits, dim=1)
```

只需要预测类别时，可以直接对 logits 使用 `argmax`，不必先 Softmax。

### 23.4 准确率

```python
accuracy = (
    predictions == labels
).float().mean()
```

逐步解释：

1. `predictions == labels` 得到布尔 tensor；
2. `.float()` 把 `True/False` 转为 `1.0/0.0`；
3. `.mean()` 计算 1 的比例，即预测正确比例。

统计整个测试集时，更严谨的方式是累加正确数量和样本总数：

```python
total_correct += (
    predictions == labels
).sum().item()

total_samples += labels.size(0)
```

最后：

```python
accuracy = total_correct / total_samples
```

---

## 24. MNIST tensor shape

一个 batch 的 MNIST 图片：

```text
images.shape = [64, 1, 28, 28]
```

含义：

- 64：图片数量；
- 1：灰度通道；
- 第一个 28：图片高度；
- 第二个 28：图片宽度。

其他关键 shape：

```text
images       [batch, 1, 28, 28]
labels       [batch]
logits       [batch, 10]
predictions  [batch]
loss         []  # 标量 tensor
```

---

## 25. 神经网络层与 `nn.Linear`

适合入门的 MNIST 模型：

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

shape 流程：

```text
[batch, 1, 28, 28]
→ Flatten
[batch, 784]
→ Linear(784, 128)
[batch, 128]
→ ReLU
[batch, 128]
→ Linear(128, 10)
[batch, 10]
```

### 25.1 各层作用

- `Flatten`：把每张 `1×28×28` 图片展平为 784 个像素，不混合不同样本；
- `Linear`：执行 `y = Wx + b`，有可训练权重和偏置；
- `ReLU`：执行 `max(0, x)`，引入非线性，不改变 shape；
- 隐藏层：位于输入和输出之间，学习中间特征；
- 输出层：产生每个类别的 logits。

如果多个线性层之间没有非线性激活，它们仍可以合并成一个线性变换，所以隐藏线性层之间通常加入 ReLU 等激活函数。

### 25.2 `nn.Linear(in_features, out_features)`

例如：

```python
nn.Linear(784, 128)
```

表示 784 个输入特征映射为 128 个输出特征。每个输出神经元都连接全部 784 个输入，因此权重数：

\[
784\times128=100352
\]

另有 128 个偏置，总参数：

\[
784\times128+128=100480
\]

通用参数数量：

\[
\boxed{
in\_features\times out\_features+out\_features
}
\]

PyTorch 的权重 shape：

```text
weight.shape = [out_features, in_features]
bias.shape   = [out_features]
```

### 25.3 多层数字如何选择

例如：

```text
784 → 256 → 128 → 10
```

规则：

- 第一个 784 由 MNIST 的 `28×28` 像素决定；
- 最后一个 10 由类别数量决定；
- 中间的 256、128 是超参数，需要用验证集选择；
- 前一层输出必须等于后一层输入。

隐藏层更宽、更深，表达能力和参数量通常更大，但计算成本和过拟合风险也更高。初学时建议先从 `784 → 128 → 10` 开始。

---

## 26. 交叉熵、NLL 与准确率

### 26.1 交叉熵

单个样本可以直观理解为：

\[
\boxed{L=-\log(p_{正确类别})}
\]

- 正确类别概率高：loss 小；
- 正确类别概率低：loss 大；
- 对错误且非常自信的预测惩罚很大。

PyTorch：

```python
loss_function = nn.CrossEntropyLoss()
logits = model(images)
loss = loss_function(logits, labels)
```

重要规则：`CrossEntropyLoss` 直接接收原始 logits，不要自己提前 Softmax。

输入要求：

```text
logits.shape = [batch_size, num_classes]
labels.shape = [batch_size]
labels.dtype = torch.long  # 通常是 int64
```

### 26.2 Negative Log Likelihood

NLL 中文为“负对数似然”：

\[
L=-\log(p_{正确类别})
\]

PyTorch 中可以理解为：

\[
\boxed{
\text{CrossEntropyLoss}
=
\text{LogSoftmax}
+
\text{NLLLoss}
}
\]

拆开写：

```python
log_probabilities = F.log_softmax(
    logits,
    dim=1
)

loss = nn.NLLLoss()(
    log_probabilities,
    labels
)
```

`NLLLoss` 期望接收对数概率，不应直接传入原始 logits。

### 26.3 交叉熵与准确率的分工

- 交叉熵：连续可导，用于训练；
- 准确率：统计最终分对多少，用于评价。

准确率暂时不变时，交叉熵仍可能下降，例如正确类别概率从 60% 提高到 90%。

---

## 27. 已经学到的 PyTorch 训练主线

目前已经能理解下面的训练流程：

```python
for epoch in range(num_epochs):
    model.train()

    for images, labels in train_loader:
        logits = model(images)
        loss = loss_function(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

逐步含义：

1. DataLoader 提供一个 batch；
2. 模型前向传播，输出 logits；
3. 交叉熵比较 logits 与正确标签，得到标量 loss；
4. `zero_grad()` 清空旧梯度；
5. `backward()` 根据链式法则计算梯度；
6. `step()` 更新模型参数；
7. 处理完所有 batch 后完成一个 epoch。

测试流程：

```python
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        logits = model(images)
        predictions = logits.argmax(dim=1)
```

测试时不反向传播、不更新参数。

---

# 第六部分：交接给下一位辅导者

## 28. 建议从哪里继续

下一步不要再增加大量新概念，应把目前所有概念串成一个完整、可运行的 MNIST 程序。

推荐顺序：

1. 检查并配置 PyTorch、torchvision 环境；
2. 下载并查看 MNIST 单条样本；
3. 打印 `image.shape`、`label`、像素范围；
4. 建立官方 train/test Dataset；
5. 从训练集划出 validation；
6. 建立 DataLoader，打印第一个 batch 的 shape；
7. 定义简单模型 `784 → 128 → 10`；
8. 逐层打印 shape；
9. 定义 `CrossEntropyLoss` 和 optimizer；
10. 手工走通一个 batch 的前向、loss、反向、更新；
11. 再写 epoch 训练循环；
12. 写 validation/test 评价循环；
13. 记录 train/validation loss 和 accuracy；
14. 保存最佳模型；
15. 显示若干测试图片、真实标签和预测标签；
16. 最后写《我理解的 PyTorch 训练流程》。

在开始完整训练前，建议让学习者先预测以下 shape：

```text
images       [64, 1, 28, 28]
flattened    [64, 784]
hidden       [64, 128]
logits       [64, 10]
labels       [64]
predictions  [64]
loss         []
```

## 29. 尚未完成的任务

- 第 1～2 周代码作业已经提交到 GitHub；后续可补强 README、依赖清单和自动化运行检查；
- 第 3 周三个完整小程序尚未在当前工作区落盘和验证；
- MNIST 完整训练程序尚未编写；
- 《我理解的 PyTorch 训练流程》尚未成文；
- CS285 指定内容尚未开始系统讲解。

## 30. CS285 的使用边界

只需要看：

- Introduction；
- Imitation Learning；
- RL Basics。

学习目标：

- 理解状态、动作、奖励、策略、轨迹等概念；
- 理解模仿学习和强化学习解决什么问题；
- 能把监督学习与行为克隆联系起来；
- 不要求深挖数学推导或完整证明。

---

# 第七部分：快速复习清单

## 31. 应该能回答的问题

### Python、NumPy 与第 1 周信号任务

1. 变量、函数、list、dict 和 class 分别适合解决什么问题？
2. NumPy array 的 `shape` 表示什么？
3. `*` 和 `@` 在 NumPy 中有什么区别？
4. 怎样用 `np.linspace` 生成时间轴？
5. `rng.normal(loc, scale, size)` 的三个主要参数是什么？
6. moving average 为什么能减弱噪声？窗口变大有什么代价？
7. NumPy Softmax 为什么先减去最大值？
8. `np.argmax` 返回的是最大值还是最大值的位置？
9. matplotlib 中 `plot`、`scatter`、`legend` 和 `savefig` 分别做什么？
10. Git 的 `status → add → commit → push` 分别表示什么？

### PyTorch 与第 2 周训练任务

1. Tensor 的 `shape`、`ndim`、`reshape` 和 `squeeze` 分别做什么？
2. 为什么 `x.reshape(-1, 1)` 能把 100 个数变为 `[100, 1]`？
3. `nn.Linear(1, 1)` 中的两个 1 分别是什么？
4. `MSELoss` 衡量什么？
5. optimizer 的工作是什么？SGD 和 Adam 都属于什么？
6. 为什么训练循环每轮都要 `zero_grad()`？
7. `backward()` 和 `step()` 的职责有什么区别？
8. `model.eval()` 和 `torch.no_grad()` 的作用有什么区别？
9. MLP 为什么需要 Tanh 或 ReLU 等非线性激活？
10. 第 2 周 MLP 的 `1 → 32 → 32 → 1` 每个数字表示什么？

### 数学与坐标变换

1. `*` 和 `@` 有什么区别？
2. 为什么局部点到世界点是 `R @ p + t`？
3. 为什么批量行向量点使用 `points @ R.T`？
4. 旋转再平移和平移再旋转为什么不同？
5. 齐次坐标解决了什么问题？

### 梯度下降

1. 导数、偏导数和梯度分别是什么？
2. 为什么梯度下降要减去梯度？
3. 学习率太大或太小会怎样？
4. MSE 为什么要把误差平方？
5. epsilon 和 learning rate 有什么区别？
6. `loss.backward()` 做什么，为什么还需要 `optimizer.step()`？

### 概率、采样与滤波

1. 均值、方差和标准差分别描述什么？
2. `torch.randn()` 默认生成什么分布？
3. 采样频率和采样周期有什么关系？
4. 什么是混叠？
5. 为什么低通滤波越平滑，通常延迟越大？

### 机器学习与 MNIST

1. 监督学习中的输入和标签是什么？
2. train、validation、test 各自负责什么？
3. Dataset 和 DataLoader 有什么区别？
4. batch、iteration、epoch 的关系是什么？
5. 过拟合有哪些典型表现？
6. `[64, 1, 28, 28]` 的每个维度是什么？
7. logits 为什么不是概率？
8. 为什么分类预测使用 `argmax(dim=1)`？
9. `nn.Linear(784, 128)` 的参数量怎样计算？
10. 为什么多个 Linear 之间需要 ReLU 等非线性？
11. 为什么 `CrossEntropyLoss` 前不要手工 Softmax？
12. `predictions == labels` 后的 `.float().mean()` 为什么是准确率？

## 32. 最小公式表

```text
坐标变换：          p_world = R @ p_robot + t
批量坐标变换：      P_world = P_robot @ R.T + t
线性模型：          y_pred = w*x + b
MSE：               mean((y_pred - y)^2)
梯度下降：          parameter -= learning_rate * gradient
数值梯度：          (L(w+eps)-L(w-eps))/(2*eps)
采样关系：          fs = 1 / Ts
低通滤波：          y[k] = y[k-1] + alpha*(x[k]-y[k-1])
每epoch的iteration：ceil(dataset_size / batch_size)
线性层参数量：      in_features*out_features + out_features
分类预测：          logits.argmax(dim=1)
交叉熵直觉：        -log(p_correct)
准确率：            correct / total
```

---

## 33. 当前阶段结论

第 1 周已经完成 NumPy 传感器信号、噪声、移动平均、Softmax、二维旋转矩阵和 GitHub 仓库；第 2 周已经完成 PyTorch 线性回归和 MLP 非线性回归；第 3 周的数学知识已经系统讲解，但三个综合程序还需落盘。知识层面已经具备开始 MNIST 完整训练的基础：tensor shape、矩阵运算、梯度下降、autograd、Dataset/DataLoader、batch/iteration/epoch、分类、logits、网络层、交叉熵和准确率都已讲过。

下一阶段的重点不是继续堆概念，而是完成一次端到端训练，并能够独立解释：

```text
数据从哪里来
→ 如何组成 batch
→ 每层 shape 怎样变化
→ 如何得到 logits 和 loss
→ backward 如何计算梯度
→ optimizer 如何更新参数
→ 如何在验证集和测试集上评价
```
