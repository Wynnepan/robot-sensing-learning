import numpy as np
import matplotlib.pyplot as plt


def moving_average(signal, window_size):
    """
    对一维信号进行移动平均滤波。

    参数:
        signal: 一维 NumPy 数组
        window_size: 移动窗口包含的数据点数量

    返回:
        滤波后的一维 NumPy 数组
    """
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0")

    if window_size > len(signal):
        raise ValueError("window_size cannot be larger than signal length")

    # 创建平均滤波核
    kernel = np.ones(window_size) / window_size

    # 将滤波核沿着信号移动
    filtered_signal = np.convolve(
        signal,
        kernel,
        mode="same",
    )

    return filtered_signal


# 1. 创建时间轴
t = np.linspace(0, 10, 1000)

# 2. 创建干净信号
frequency = 1.0
clean_signal = np.sin(2 * np.pi * frequency * t)

# 3. 添加噪声
rng = np.random.default_rng(seed=42)
noise = rng.normal(
    loc=0.0,
    scale=0.3,
    size=t.shape,
)
noisy_signal = clean_signal + noise

# 4. 使用移动平均滤波器
window_size = 20
filtered_signal = moving_average(
    noisy_signal,
    window_size,
)

# 5. 输出数组信息，检查长度是否一致
print("Time shape:", t.shape)
print("Noisy signal shape:", noisy_signal.shape)
print("Filtered signal shape:", filtered_signal.shape)

# 6. 画图
plt.figure(figsize=(12, 6))

plt.plot(
    t,
    noisy_signal,
    color="gray",
    alpha=0.45,
    label="Noisy signal",
)

plt.plot(
    t,
    clean_signal,
    color="blue",
    linewidth=2,
    label="Clean signal",
)

plt.plot(
    t,
    filtered_signal,
    color="red",
    linewidth=2,
    label=f"Moving average (window={window_size})",
)

plt.title("Moving Average Filter")
plt.xlabel("Time (s)")
plt.ylabel("Sensor Value")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("week1/moving_average.png", dpi=150)
print("Saved: week1/moving_average.png")