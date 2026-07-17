import torch
from torch import nn

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


# 生成训练输入
# ====================== 1. 造数据 ======================
# 100个输入点
x = torch.linspace(-5, 5, 100)
# 将 x 重塑为一个列向量
x = x.reshape(-1, 1)
print("After reshape:", x.shape)
print()

# 真实世界公式：y = 3x + 2（我们让模型去学这个 3 和 2）
y_real = 3 * x + 2  

# ====================== 2. 搭建模型 ======================
model = nn.Linear(1, 1)  # 输入1个，输出1个：y = w*x + b

predictions = model(x)

# print("Model:")
# print(model)
# print("weight:", model.weight)
# print("bias:", model.bias)
# print("input shape:", x.shape)
# print("output shape:", predictions.shape)
# print("first five predictions:")
# print(predictions[:5])

# ====================== 3. 定义训练工具 ======================
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 优化器（更新w/b）
loss_fn = nn.MSELoss()                                   # 损失函数（算误差）

# ====================== 4. 训练开始（循环500轮） ==================
print("===== 训练前（随机初始化的 w 和 b）=====")
print("weight (w):", model.weight.item())
print("bias (b):", model.bias.item())
print("---------------------------------------")

for epoch in range(500):
    # 1️⃣ 前向传播：算预测值 y'
    y_pred = model(x)

    # 2️⃣ 算误差：预测值 和 真实值 差多少
    loss = loss_fn(y_pred, y_real)

    # 3️⃣ 清空上次梯度（必须加！）
    optimizer.zero_grad()

    # 4️⃣ 反向传播：算 w 和 b 的梯度
    loss.backward()

    # 5️⃣ 更新权重：让 w、b 变更好
    optimizer.step()

    # 每100轮打印一次进度
    if epoch % 100 == 0:
        print(f"轮次 {epoch} | 损失: {loss.item():.4f}")

# ====================== 5. 训练结束看结果 ======================
print("\n===== 训练完成！学到的 w 和 b =====")
print("weight (w):", model.weight.item())
print("bias (b):", model.bias.item())
print("\n真实应该是：w=3，b=2")