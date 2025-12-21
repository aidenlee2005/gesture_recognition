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

def visualize_gesture_hand_keypoints_3d():
    """可视化不同手势的手部关键点3D立体效果 - 分开并排显示"""
    # 加载数据
    data = np.load('hand_gesture_data.npz')
    X = data['X']
    y = data['y']

    # 计算子图布局
    n_gestures = len(GESTURES)
    n_cols = min(3, n_gestures)  # 最多3列
    n_rows = (n_gestures + n_cols - 1) // n_cols  # 计算需要的行数

    fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
    fig.suptitle('Hand Keypoints 3D Visualization by Gesture', fontsize=16, fontweight='bold')

    # 为每个手势创建子图
    for idx, (gesture_id, gesture_name) in enumerate(GESTURES.items()):
        # 找到该手势的样本
        gesture_indices = np.where(y == gesture_id)[0]
        if len(gesture_indices) == 0:
            print(f"警告: 手势 '{gesture_name}' 没有样本")
            continue

        # 随机选择一个样本
        sample_idx = np.random.choice(gesture_indices)
        sample = X[sample_idx]

        # 选择序列中间的帧
        frame_idx = len(sample) // 2
        frame = sample[frame_idx]

        # 提取手部关键点
        # subtask1数据格式: 每帧63维 = 21个手部关键点 × 3坐标
        hand_points = frame.reshape(21, 3)  # 21个手部关键点

        # 只保留有效的关键点 (非零值)
        valid_mask = np.any(hand_points != 0, axis=1)
        valid_hand_points = hand_points[valid_mask]

        if len(valid_hand_points) == 0:
            print(f"警告: 手势 '{gesture_name}' 的样本没有有效的手部关键点")
            continue

        # 创建子图
        ax = fig.add_subplot(n_rows, n_cols, idx+1, projection='3d')

        # 计算该手势关键点的范围，用于居中显示
        min_vals = np.min(valid_hand_points, axis=0)
        max_vals = np.max(valid_hand_points, axis=0)
        center = (min_vals + max_vals) / 2
        ranges = max_vals - min_vals

        # 设置边距
        margin = 0.2
        axis_limits = []
        for i in range(3):
            axis_range = ranges[i] * (1 + margin)
            axis_min = center[i] - axis_range / 2
            axis_max = center[i] + axis_range / 2
            axis_limits.append((axis_min, axis_max))

        # 绘制关键点
        ax.scatter(valid_hand_points[:, 0], valid_hand_points[:, 1], valid_hand_points[:, 2],
                  c='red', marker='o', s=80, alpha=0.8)

        # 绘制手部连接线 (只对有效的关键点)
        for connection in HAND_CONNECTIONS:
            start, end = connection
            if start < len(hand_points) and end < len(hand_points):
                # 检查两个点是否都有效
                if valid_mask[start] and valid_mask[end]:
                    ax.plot([hand_points[start, 0], hand_points[end, 0]],
                           [hand_points[start, 1], hand_points[end, 1]],
                           [hand_points[start, 2], hand_points[end, 2]],
                           'blue', alpha=0.7, linewidth=3)

        # 设置坐标轴
        ax.set_xlim(axis_limits[0])
        ax.set_ylim(axis_limits[1])
        ax.set_zlim(axis_limits[2])

        # 设置等比例缩放
        ax.set_box_aspect([1, 1, 1])

        # 设置标签和标题
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.set_title(f'{gesture_name}', fontsize=14, fontweight='bold', pad=20)

        # 设置视角
        ax.view_init(elev=25, azim=135)

        # 添加网格
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

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

    # 生成手势手部关键点3D对比图
    print("\n🔲 生成不同手势手部关键点3D对比可视化...")
    fig = visualize_gesture_hand_keypoints_3d()
    fig.savefig('gesture_hand_keypoints_3d.png', dpi=150, bbox_inches='tight')
    print("✅ 手势手部关键点3D对比图已保存: gesture_hand_keypoints_3d.png")

    # 随机可视化5个样本
    visualize_random_samples(X, y, num_samples=5)

    # 可视化第一个样本的第一帧
    visualize_hand_landmarks(X, 0)