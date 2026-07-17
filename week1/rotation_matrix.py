import numpy as np


def rotation_matrix_2d(angle_degrees):
    """
    创建二维逆时针旋转矩阵。

    参数:
        angle_degrees: 旋转角度，单位为度

    返回:
        形状为 (2, 2) 的 NumPy 数组
    """
    # NumPy 的 sin 和 cos 使用弧度，因此先把角度转换成弧度
    angle_radians = np.deg2rad(angle_degrees)

    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)

    matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta],
    ])

    return matrix


# 创建一个二维点：(1, 0)
point = np.array([1.0, 0.0])

# 创建逆时针旋转 90 度的矩阵
rotation = rotation_matrix_2d(90)

# 使用矩阵乘法旋转这个点
rotated_point = rotation @ point

print("Rotation matrix:")
print(rotation)

print("\nOriginal point:")
print(point)

print("\nRotated point:")
print(rotated_point)

# 把非常接近 0 的浮点误差显示成 0
print("\nRounded rotated point:")
print(np.round(rotated_point, decimals=10))


print("\nMore tests:")

test_point = np.array([1.0, 0.0])

for angle in [0, 90, 180, 270, 360, -90]:
    matrix = rotation_matrix_2d(angle)
    result = matrix @ test_point
    result = np.round(result, decimals=10)

    print(f"{angle:>4} degrees -> {result}")