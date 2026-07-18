import torch
from torch import nn
import matplotlib.pyplot as plt
# ## 基础介绍
# # 0维 Tensor
# scalar = torch.tensor(5.0)

# print("Scalar:")
# print("value:", scalar)
# print("shape:", scalar.shape)
# print("ndim:", scalar.ndim)
# print()


# # 1维 Tensor
# vector = torch.tensor([5.0])

# print("Vector:")
# print("value:", vector)
# print("shape:", vector.shape)
# print("ndim:", vector.ndim)
# print()


# # 2维 Tensor
# matrix = torch.tensor([
#     [1.0, 2.0],
#     [3.0, 4.0],
#     [5.0, 6.0],
# ])

# print("Matrix:")
# print("value:")
# print(matrix)
# print("shape:", matrix.shape)
# print("ndim:", matrix.ndim)
# print()

# 固定随机种子，让每次运行结果更容易复现
torch.manual_seed(42)

# 生成训练输入
# ====================== 1. 造数据 ======================
# 生成从 -5 到 5 的 100 个输入点
x = torch.linspace(-5, 5, 100).reshape(-1, 1)

# 正确规律：y = 3x + 2 （让模型去学这个 3 和 2）
y = 3 * x + 2
# x = torch.linspace(-5, 5, 100)
# # 将 x 重塑为一个列向量
# x = x.reshape(-1, 1)
# print("After reshape:", x.shape)
# print()

# ====================== 2. 搭建模型 ======================
# 输入1个，输出1个：y = w*x + b
model = nn.Linear(1, 1)
# print("Model:")
# print(model)
# print("weight:", model.weight)
# print("bias:", model.bias)
# print("input shape:", x.shape)
# print("output shape:", predictions.shape)
# print("first five predictions:")
# print(predictions[:5])

# ====================== 3. 定义训练工具 ======================
# 3. 创建损失函数，算误差
loss_function = nn.MSELoss()
# 4. 创建优化器（更新w/b）
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
)

# ====================== 4. 训练开始（循环500轮） ==================
print("===== 训练前（随机初始化的 w 和 b）=====")
print("weight (w):", model.weight.item())
print("bias (b):", model.bias.item())
print("---------------------------------------")

epochs = 1000

for epoch in range(epochs):

    # 1️⃣ 清空上次梯度
    optimizer.zero_grad()

    # 2️⃣ 前向传播：算预测值 y'
    predictions = model(x)

    # 3️⃣ 算误差：预测值 和 真实值 差多少
    loss = loss_function(predictions, y)

    # 4️⃣ 反向传播：算 w 和 b 的梯度
    loss.backward()

    # 5️⃣ 更新参数权重：让 w、b 变更好
    optimizer.step()

    # 每100轮打印一次进度
    if (epoch + 1) % 100 == 0:
        print(
            f"轮次 {epoch + 1:4d} |" 
            f"损失: {loss.item():.5f}"
            )

# ====================== 5. 训练结束看结果 ======================
learned_weight = model.weight.item()
learned_bias = model.bias.item()
print("\n===== 训练完成！学到的 w 和 b =====")
print("\nLearned result:")
print(f"weight = {learned_weight:.4f}")
print(f"bias   = {learned_bias:.4f}")
print("\n真实应该是：w=3，b=2")

# ====================== 6. 训练结束后生成最终预测 ======================
# no_grad 表示这里只预测，不需要记录梯度
with torch.no_grad():
    final_predictions = model(x)

# ====================== 7. 8. 转成 NumPy，交给 Matplotlib 画图 ======================
x_numpy = x.squeeze().numpy()
y_numpy = y.squeeze().numpy()
prediction_numpy = final_predictions.squeeze().numpy()


plt.figure(figsize=(10, 6))

plt.scatter(
    x_numpy,
    y_numpy,
    s=20,
    label="Training data",
)

plt.plot(
    x_numpy,
    prediction_numpy,
    color="red",
    linewidth=2,
    label="Model prediction",
)

plt.title("PyTorch: Fit y = 3x + 2")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("week2/linear_regression.png", dpi=150)
plt.close()

print("Saved: week2/linear_regression.png")