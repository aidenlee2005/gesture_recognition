import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# MediaPipe 手部关键点连接
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
]

# 手势类别名称
GESTURES = {
    0: "OK",
    1: "Thumbs Up",
    2: "Yeah",
    3: "Fist",
    4: "Palm",
}

def visualize_hand_landmarks(landmarks, title="Hand Landmarks", save_path=None):
    """可视化单个帧的手部关键点"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制关键点
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c='r', marker='o', s=50)

    # 绘制连接
    for connection in HAND_CONNECTIONS:
        start, end = connection
        ax.plot([landmarks[start, 0], landmarks[end, 0]],
                [landmarks[start, 1], landmarks[end, 1]],
                [landmarks[start, 2], landmarks[end, 2]], 'b-', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_xlim([-0.2, 1.2])
    ax.set_ylim([-0.2, 1.2])
    ax.set_zlim([-0.2, 1.2])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()
    plt.close(fig)

def visualize_random_samples(X, y, num_samples=5):
    """随机可视化多个样本"""
    total_samples = len(X)
    if num_samples > total_samples:
        num_samples = total_samples

    # 随机选择样本
    sample_indices = random.sample(range(total_samples), num_samples)

    for idx in sample_indices:
        # 选择序列中间的帧
        frame_idx = len(X[idx]) // 2
        landmarks = X[idx, frame_idx].reshape(21, 3)
        gesture_label = y[idx]
        gesture_name = GESTURES.get(gesture_label, f"Unknown ({gesture_label})")

        print(f"可视化样本 {idx}: 手势 '{gesture_name}' (标签 {gesture_label})")
        visualize_hand_landmarks(landmarks, f"Sample {idx} - {gesture_name}")

if __name__ == "__main__":
    # 加载数据
    data = np.load('hand_gesture_data.npz')
    X = data['X']
    y = data['y']

    print(f"数据集信息:")
    print(f"总样本数: {len(X)}")
    print(f"每个样本序列长度: {X.shape[1]} 帧")
    print(f"每帧特征维度: {X.shape[2]}")
    print(f"标签分布: {np.bincount(y)}")

    # 随机可视化5个样本
    visualize_random_samples(X, y, num_samples=5)

    # 可视化第一个样本的第一帧
    visualize_hand_landmarks(X, 0)