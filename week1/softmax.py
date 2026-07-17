import numpy as np


def softmax(x):
    """
    使用 NumPy 计算一维数组的 softmax。

    参数:
        x: 一维数组，例如 [2.0, 1.0, 0.1]

    返回:
        概率数组，所有元素之和为 1
    """
    # 将输入转换为浮点 NumPy 数组
    x = np.asarray(x, dtype=float)

    # 本练习先只处理一维数组
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array")

    # 防止指数运算发生数值溢出
    shifted_x = x - np.max(x)

    # 对每个元素计算指数 e^x
    exp_x = np.exp(shifted_x)

    # 每个指数除以所有指数之和
    probabilities = exp_x / np.sum(exp_x)

    return probabilities


# 测试
scores = np.array([2.0, 1.0, 0.1])
probabilities = softmax(scores)

print("Original scores:", scores)
print("Softmax probabilities:", probabilities)
print("Probability sum:", np.sum(probabilities))
print("Largest class index:", np.argmax(probabilities))