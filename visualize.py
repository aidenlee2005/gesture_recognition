import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# MediaPipe 手部关键点连接
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
]

def visualize_hand_landmarks(sequence, sample_idx=0, frame_idx=0):
    """可视化手部关键点"""
    if sample_idx >= len(sequence):
        print("样本索引超出范围")
        return

    landmarks = sequence[sample_idx, frame_idx]  # (63,)
    # 重塑为 (21, 3)
    landmarks = landmarks.reshape(21, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制关键点
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c='r', marker='o')

    # 绘制连接
    for connection in HAND_CONNECTIONS:
        start, end = connection
        ax.plot([landmarks[start, 0], landmarks[end, 0]],
                [landmarks[start, 1], landmarks[end, 1]],
                [landmarks[start, 2], landmarks[end, 2]], 'b-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Hand Landmarks - Sample {sample_idx}')
    plt.show()

if __name__ == "__main__":
    # 加载数据
    data = np.load('hand_gesture_data.npz')
    X = data['X']
    y = data['y']
    print(f"数据形状: {X.shape}")
    print(f"标签: {y}")

    # 可视化第一个样本的第一帧
    visualize_hand_landmarks(X, 0)