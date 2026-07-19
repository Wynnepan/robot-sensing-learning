# Multi-Layer Perceptron （多层感知机，MLP）任务  拟合这个函数：y=sin(x)+噪声
import torch
from torch import nn
import matplotlib.pyplot as plt

# # ============= MLP模型的定义 =============
# # MLP 只定义模型结构
### ================== 1、造数据 ======================
# 固定随机种子，让每次运行结果更容易复现，生成从 -5 到 5 的 100 个输入点，并reshape为列向量
torch.manual_seed(42)
x = torch.linspace(-5, 5, 100).reshape(-1, 1)
# 正确规律：y = 3x + 2 （让模型去学这个 3 和 2）
y_clean = torch.sin(x)
noise = 0.1 * torch.randn_like(y_clean)  # 添加噪声
y_noisy = y_clean + noise  # 带噪声的目标值

# layer1 = nn.Linear(1, 32)
# activation1 = nn.Tanh()
# layer2 = nn.Linear(32, 32)
# activation2 = nn.Tanh()
# layer3 = nn.Linear(32, 1)

# print("原始输入:", x.shape)

# x1 = layer1(x)
# print("第一层 Linear 后:", x1.shape)

# x2 = activation1(x1)
# print("第一次 Tanh 后:", x2.shape)

# x3 = layer2(x2)
# print("第二层 Linear 后:", x3.shape)

# x4 = activation2(x3)
# print("第二次 Tanh 后:", x4.shape)

# prediction = layer3(x4)
# print("最终输出:", prediction.shape)


# ====================== 2. 搭建模型 ======================
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)

# ====================== 3. 定义训练工具 ======================
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
)

# ====================== 4. 训练开始（循环500轮） ===============
epochs = 2000
for epoch in range(epochs):
    # 1️⃣ 清空上次梯度
    optimizer.zero_grad()
    # 2️⃣ 前向传播：算预测值 y'
    predictions = model(x)
    # 3️⃣ 算误差：预测值 和 真实值 差多少
    loss = loss_function(predictions, y_noisy)
    # 4️⃣ 反向传播：算梯度
    loss.backward()
    # 5️⃣ 更新参数权重
    optimizer.step()

      # 每100轮打印一次进度
    if (epoch + 1) % 200 == 0:
        print(
            f"轮次 {epoch + 1:4d} |" 
            f"损失: {loss.item():.5f}"
            )
        
# ====================== 5. 训练结束看结果 ======================
model.eval()

with torch.no_grad():
    predictions = model(x)
x_numpy = x.squeeze().numpy()
y_clean_numpy = y_clean.squeeze().numpy()
y_noisy_numpy = y_noisy.squeeze().numpy()
prediction_numpy = predictions.squeeze().numpy()

plt.figure(figsize=(10, 6))

# 带噪声的数据点
plt.scatter(
    x_numpy,
    y_noisy_numpy,
    s=15,
    color="gray",
    alpha=0.5,
    label="Noisy data",
)

# 没有噪声的真实函数
plt.plot(
    x_numpy,
    y_clean_numpy,
    color="blue",
    linewidth=2,
    label="True sin(x)",
)

# MLP学到的函数
plt.plot(
    x_numpy,
    prediction_numpy,
    color="red",
    linewidth=2,
    label="MLP prediction",
)

plt.title("MLP Fits a Noisy Nonlinear Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("week2/mlp_noisy_function.png", dpi=150)
plt.show()
plt.close()

print("Saved: week2/mlp_noisy_function.png")
print(model)