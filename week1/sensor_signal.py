import numpy as np
import matplotlib.pyplot as plt


# 1. 生成时间轴
# 从 0 秒到 10 秒，均匀取 1000 个时间点
t = np.linspace(0, 10, 1000)

# 2. 生成干净的模拟传感器信号
# 这里用频率为 1 Hz 的正弦波模拟传感器信号
frequency = 1.0
clean_signal = np.sin(2 * np.pi * frequency * t)

# 3. 生成随机噪声
rng = np.random.default_rng(seed=42)
noise = rng.normal(
    loc=0.0,
    scale=0.3,
    size=t.shape,
)

# 4. 把噪声添加到干净信号中
noisy_signal = clean_signal + noise

# 5. 画图
plt.figure(figsize=(12, 6))

plt.plot(
    t,
    clean_signal,
    label="Clean signal",
    color="blue",
    linewidth=2,
)

plt.plot(
    t,
    noisy_signal,
    label="Noisy signal",
    color="gray",
    alpha=0.6,
)

plt.title("Simulated Sensor Signal")
plt.xlabel("Time (s)")
plt.ylabel("Sensor Value")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# 服务器通常没有图形界面，所以将图片保存到文件
plt.savefig("week1/sensor_signal.png", dpi=150)

print("Saved: week1/sensor_signal.png")