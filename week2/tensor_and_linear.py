import torch
from torch import nn


# 0维 Tensor
scalar = torch.tensor(5.0)

print("Scalar:")
print("value:", scalar)
print("shape:", scalar.shape)
print("ndim:", scalar.ndim)
print()


# 1维 Tensor
vector = torch.tensor([5.0])

print("Vector:")
print("value:", vector)
print("shape:", vector.shape)
print("ndim:", vector.ndim)
print()


# 2维 Tensor
matrix = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
])

print("Matrix:")
print("value:")
print(matrix)
print("shape:", matrix.shape)
print("ndim:", matrix.ndim)
print()


# 生成训练输入
x = torch.linspace(-5, 5, 100)
print("Before reshape:", x.shape)

x = x.reshape(-1, 1)
print("After reshape:", x.shape)
print()


# 一个输入、一个输出的线性模型
model = nn.Linear(1, 1)

predictions = model(x)

print("Model:")
print(model)
print("weight:", model.weight)
print("bias:", model.bias)
print("input shape:", x.shape)
print("output shape:", predictions.shape)
print("first five predictions:")
print(predictions[:5])