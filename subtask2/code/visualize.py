# 数据集可视化工具 - 增强前后对比分析

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置字体为英文
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# MediaPipe 手部关键点连接
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
]

# MediaPipe 姿势关键点连接（简化版）
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # 左臂
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # 右臂
    (11, 23), (12, 24), (23, 24),  # 躯干
    (23, 25), (25, 27), (27, 29), (29, 31),  # 左腿
    (24, 26), (26, 28), (28, 30), (30, 32),  # 右腿
]

# 手势类别名称
GESTURES = {
    0: "Hello", 1: "Thank you", 2: "Sorry", 3: "You", 4: "Goodbye",
    5: "I", 6: "Love", 7: "Help", 8: "Eat", 9: "Drink"
}

def load_data(data_path):
    """加载数据文件"""
    try:
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        return X, y
    except Exception as e:
        print(f"❌ 加载数据失败 {data_path}: {e}")
        return None, None

def plot_dataset_statistics(X, y, title="Dataset Statistics"):
    """绘制数据集统计信息"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. 类别分布
    class_counts = Counter(y)
    classes = [GESTURES[i] for i in sorted(class_counts.keys())]
    counts = [class_counts[i] for i in sorted(class_counts.keys())]

    axes[0, 0].bar(classes, counts, color='skyblue', alpha=0.8)
    axes[0, 0].set_title('Class Distribution', fontweight='bold')
    axes[0, 0].set_ylabel('Sample Count')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 添加数值标签
    for i, v in enumerate(counts):
        axes[0, 0].text(i, v + max(counts) * 0.01, str(v), ha='center', fontweight='bold')

    # 2. 特征维度分布
    feature_means = np.mean(X.reshape(X.shape[0], -1), axis=0)
    feature_stds = np.std(X.reshape(X.shape[0], -1), axis=0)

    axes[0, 1].hist(feature_means, bins=50, alpha=0.7, color='lightcoral', label='Mean')
    axes[0, 1].hist(feature_stds, bins=50, alpha=0.7, color='lightgreen', label='Std Dev')
    axes[0, 1].set_title('Feature Distribution Statistics', fontweight='bold')
    axes[0, 1].set_xlabel('Feature Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # 3. Sequence length distribution (if applicable)
    if len(X.shape) == 3:  # (samples, seq_len, features)
        seq_lengths = [np.sum(np.any(frame != 0, axis=1)) for frame in X]
        axes[1, 0].hist(seq_lengths, bins=30, alpha=0.8, color='gold')
        axes[1, 0].set_title('Sequence Valid Length Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Valid Frames')
        axes[1, 0].set_ylabel('Frequency')
    else:
        # If 2D data, show other statistics
        non_zero_ratios = np.mean(X != 0, axis=1)
        axes[1, 0].hist(non_zero_ratios, bins=30, alpha=0.8, color='gold')
        axes[1, 0].set_title('Non-zero Feature Ratio Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Non-zero Ratio')
        axes[1, 0].set_ylabel('Frequency')

    # 4. 数据质量分析
    quality_scores = []
    for sample in X:
        if len(sample.shape) == 2:  # 序列数据
            # 计算每帧的非零比例
            frame_qualities = [np.mean(frame != 0) for frame in sample]
            quality_scores.append(np.mean(frame_qualities))
        else:  # 单帧数据
            quality_scores.append(np.mean(sample != 0))

    axes[1, 1].hist(quality_scores, bins=30, alpha=0.8, color='purple')
    axes[1, 1].axvline(np.mean(quality_scores), color='red', linestyle='--',
                      label=f'Mean: {np.mean(quality_scores):.3f}')
    axes[1, 1].set_title('Data Quality Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Quality Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()

    plt.tight_layout()
    return fig

def plot_augmentation_comparison():
    """绘制数据增强前后对比"""
    # 加载原始数据和增强数据
    X_original, y_original = load_data('hand_gesture_data.npz')
    X_augmented, y_augmented = load_data('hand_gesture_data_augmented.npz')

    if X_original is None or X_augmented is None:
        print("❌ 无法加载数据文件")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Data Augmentation Before vs After Comparison', fontsize=16, fontweight='bold')

    # 1. 样本数量对比
    original_counts = Counter(y_original)
    augmented_counts = Counter(y_augmented)

    classes = [GESTURES[i] for i in sorted(original_counts.keys())]
    original_nums = [original_counts[i] for i in sorted(original_counts.keys())]
    augmented_nums = [augmented_counts[i] for i in sorted(augmented_counts.keys())]

    x = np.arange(len(classes))
    width = 0.35

    axes[0, 0].bar(x - width/2, original_nums, width, label='Original Data',
                   color='skyblue', alpha=0.8)
    axes[0, 0].bar(x + width/2, augmented_nums, width, label='Augmented Data',
                   color='lightcoral', alpha=0.8)

    axes[0, 0].set_title('Sample Count Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('Sample Count')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(classes, rotation=45)
    axes[0, 0].legend()

    # 2. 特征方差对比
    # 将数据reshape成 (samples, -1) 形式
    X_original_flat = X_original.reshape(X_original.shape[0], -1)  # (100, 6750)
    X_augmented_flat = X_augmented.reshape(X_augmented.shape[0], -1)  # (4000, 6750)

    # 计算每个特征的方差
    original_variances = np.var(X_original_flat, axis=0)  # 6750个特征的方差
    augmented_variances = np.var(X_augmented_flat, axis=0)  # 6750个特征的方差

    # 过滤掉方差为0的特征（完全不变的特征）
    valid_mask = (original_variances > 1e-6) | (augmented_variances > 1e-6)
    original_variances = original_variances[valid_mask]
    augmented_variances = augmented_variances[valid_mask]

    # 计算统计信息
    zero_var_original = np.sum(np.var(X_original_flat, axis=0) == 0)
    zero_var_augmented = np.sum(np.var(X_augmented_flat, axis=0) == 0)

    axes[0, 1].hist(original_variances, bins=50, alpha=0.7, color='skyblue',
                    label=f'Original (zero-var: {zero_var_original})', density=True)
    axes[0, 1].hist(augmented_variances, bins=50, alpha=0.7, color='lightcoral',
                    label=f'Augmented (zero-var: {zero_var_augmented})', density=True)

    axes[0, 1].set_title('Feature Variance Comparison', fontweight='bold')
    axes[0, 1].set_xlabel('Feature Variance')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_xlim(0, np.percentile(np.concatenate([original_variances, augmented_variances]), 95))
    axes[0, 1].legend()

    # 3. 数据质量对比
    def calculate_quality_scores(X):
        scores = []
        for sample in X:
            if len(sample.shape) == 2:  # 序列数据
                frame_qualities = [np.mean(frame != 0) for frame in sample]
                scores.append(np.mean(frame_qualities))
            else:
                scores.append(np.mean(sample != 0))
        return scores

    original_quality = calculate_quality_scores(X_original)
    augmented_quality = calculate_quality_scores(X_augmented)

    axes[0, 2].hist(original_quality, bins=30, alpha=0.7, color='skyblue',
                    label=f'Original (mean: {np.mean(original_quality):.3f})', density=True)
    axes[0, 2].hist(augmented_quality, bins=30, alpha=0.7, color='lightcoral',
                    label=f'Augmented (mean: {np.mean(augmented_quality):.3f})', density=True)

    axes[0, 2].set_title('Data Quality Comparison', fontweight='bold')
    axes[0, 2].set_xlabel('Quality Score')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].legend()

    # 4. 增强效果统计
    enhancement_stats = {
        'sample_amplification': len(X_augmented) / len(X_original),
        'quality_change': np.mean(augmented_quality) - np.mean(original_quality),
        'class_balance': np.std(list(augmented_counts.values())) / np.std(list(original_counts.values()))
    }

    stats_text = f"""
    Data Augmentation Effects:

    📊 Sample Count: {len(X_original)} → {len(X_augmented)}
       Amplification: {enhancement_stats['sample_amplification']:.1f}x

    🎯 Quality Change: {enhancement_stats['quality_change']:+.3f}

    ⚖️ Class Balance: {enhancement_stats['class_balance']:.2f}
       (Lower is more balanced)
    """

    axes[1, 0].text(0.1, 0.5, stats_text, transform=axes[1, 0].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    axes[1, 0].set_title('Augmentation Effects Statistics', fontweight='bold')
    axes[1, 0].axis('off')

    # 5. 单个样本对比可视化
    # 选择同一个类别的样本进行对比
    sample_class = 0  # Hello
    original_indices = np.where(y_original == sample_class)[0]
    augmented_indices = np.where(y_augmented == sample_class)[0]

    if len(original_indices) > 0 and len(augmented_indices) > 0:
        # 选择一个原始样本和一个增强样本
        original_sample = X_original[original_indices[0]]
        augmented_sample = X_augmented[augmented_indices[0]]

        # 如果是序列数据，取中间帧
        if len(original_sample.shape) == 2:
            original_frame = original_sample[len(original_sample)//2]
            augmented_frame = augmented_sample[len(augmented_sample)//2]
        else:
            original_frame = original_sample
            augmented_frame = augmented_sample

        # 绘制关键点对比
        plot_keypoints_comparison(original_frame, augmented_frame, axes[1, 1], axes[1, 2])

    plt.tight_layout()
    return fig

def plot_keypoints_comparison(original_frame, augmented_frame, ax1, ax2):
    """绘制关键点对比"""
    # 分离姿势和手势关键点
    pose_original = original_frame[:99].reshape(-1, 3)  # 33个姿势点
    hand_original = original_frame[99:].reshape(-1, 3)  # 42个手势点

    pose_augmented = augmented_frame[:99].reshape(-1, 3)
    hand_augmented = augmented_frame[99:].reshape(-1, 3)

    # Original data
    ax1.scatter(pose_original[:, 0], pose_original[:, 1], c='blue', alpha=0.6, s=15, label='Pose')
    ax1.scatter(hand_original[:, 0], hand_original[:, 1], c='red', alpha=0.8, s=20, label='Hand')
    ax1.set_title('Original Sample Keypoints', fontweight='bold')
    ax1.set_xlim(-0.2, 1.2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.legend()

    # Augmented data
    ax2.scatter(pose_augmented[:, 0], pose_augmented[:, 1], c='blue', alpha=0.6, s=15, label='Pose')
    ax2.scatter(hand_augmented[:, 0], hand_augmented[:, 1], c='red', alpha=0.8, s=20, label='Hand')
    ax2.set_title('Augmented Sample Keypoints', fontweight='bold')
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.legend()

def visualize_3d_keypoints_comparison():
    """3D关键点立体可视化对比"""
    X_original, y_original = load_data('hand_gesture_data.npz')
    X_augmented, y_augmented = load_data('hand_gesture_data_augmented.npz')

    if X_original is None or X_augmented is None:
        print("❌ Unable to load data files")
        return

    # 选择同一个类别的样本进行对比
    sample_class = 0  # Hello
    original_indices = np.where(y_original == sample_class)[0]
    augmented_indices = np.where(y_augmented == sample_class)[0]

    if len(original_indices) == 0 or len(augmented_indices) == 0:
        print("❌ No samples found for the selected class")
        return

    # 随机选择样本
    original_sample = X_original[np.random.choice(original_indices)]
    augmented_sample = X_augmented[np.random.choice(augmented_indices)]

    # 如果是序列数据，取中间帧
    if len(original_sample.shape) == 2:
        original_frame = original_sample[len(original_sample)//2]
        augmented_frame = augmented_sample[len(augmented_sample)//2]
    else:
        original_frame = original_sample
        augmented_frame = augmented_sample

    # 分离姿势和手势关键点
    pose_original = original_frame[:99].reshape(-1, 3)  # 33 pose points
    hand_original = original_frame[99:].reshape(-1, 3)  # 42 hand points

    pose_augmented = augmented_frame[:99].reshape(-1, 3)
    hand_augmented = augmented_frame[99:].reshape(-1, 3)

    # 计算所有关键点的范围，用于设置坐标轴
    all_points_original = np.vstack([pose_original, hand_original])
    all_points_augmented = np.vstack([pose_augmented, hand_augmented])

    # 计算联合范围
    all_points = np.vstack([all_points_original, all_points_augmented])

    # 只考虑非零点（有效的关键点）
    valid_points = all_points[np.any(all_points != 0, axis=1)]

    if len(valid_points) == 0:
        print("❌ No valid keypoints found")
        return

    # 计算范围和中心
    min_vals = np.min(valid_points, axis=0)
    max_vals = np.max(valid_points, axis=0)
    center = (min_vals + max_vals) / 2
    ranges = max_vals - min_vals

    # 设置合适的边界（增加10%的边距）
    margin = 0.1
    axis_limits = []
    for i in range(3):
        axis_range = ranges[i] * (1 + margin)
        axis_min = center[i] - axis_range / 2
        axis_max = center[i] + axis_range / 2
        axis_limits.append((axis_min, axis_max))

    # 创建3D图形
    fig = plt.figure(figsize=(20, 10))

    # 原始数据3D图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pose_original[:, 0], pose_original[:, 1], pose_original[:, 2],
               c='blue', alpha=0.7, s=30, label='Pose Keypoints')
    ax1.scatter(hand_original[:, 0], hand_original[:, 1], hand_original[:, 2],
               c='red', alpha=0.8, s=40, label='Hand Keypoints')

    # 绘制姿势连接线
    for connection in POSE_CONNECTIONS:
        start, end = connection
        if start < len(pose_original) and end < len(pose_original):
            # 只绘制非零点之间的连接
            if np.any(pose_original[start] != 0) and np.any(pose_original[end] != 0):
                ax1.plot([pose_original[start, 0], pose_original[end, 0]],
                        [pose_original[start, 1], pose_original[end, 1]],
                        [pose_original[start, 2], pose_original[end, 2]],
                        'blue', alpha=0.6, linewidth=2)

    # 绘制手势连接线
    for connection in HAND_CONNECTIONS:
        start, end = connection
        if start < len(hand_original) and end < len(hand_original):
            # 只绘制非零点之间的连接
            if np.any(hand_original[start] != 0) and np.any(hand_original[end] != 0):
                ax1.plot([hand_original[start, 0], hand_original[end, 0]],
                        [hand_original[start, 1], hand_original[end, 1]],
                        [hand_original[start, 2], hand_original[end, 2]],
                        'red', alpha=0.7, linewidth=2)

    ax1.set_title('Original Data - 3D Keypoints with Connections', fontweight='bold', fontsize=14)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Z Coordinate')
    ax1.set_xlim(axis_limits[0])
    ax1.set_ylim(axis_limits[1])
    ax1.set_zlim(axis_limits[2])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 增强数据3D图
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(pose_augmented[:, 0], pose_augmented[:, 1], pose_augmented[:, 2],
               c='cyan', alpha=0.7, s=30, label='Pose Keypoints')
    ax2.scatter(hand_augmented[:, 0], hand_augmented[:, 1], hand_augmented[:, 2],
               c='orange', alpha=0.8, s=40, label='Hand Keypoints')

    # 绘制姿势连接线
    for connection in POSE_CONNECTIONS:
        start, end = connection
        if start < len(pose_augmented) and end < len(pose_augmented):
            # 只绘制非零点之间的连接
            if np.any(pose_augmented[start] != 0) and np.any(pose_augmented[end] != 0):
                ax2.plot([pose_augmented[start, 0], pose_augmented[end, 0]],
                        [pose_augmented[start, 1], pose_augmented[end, 1]],
                        [pose_augmented[start, 2], pose_augmented[end, 2]],
                        'cyan', alpha=0.6, linewidth=2)

    # 绘制手势连接线
    for connection in HAND_CONNECTIONS:
        start, end = connection
        if start < len(hand_augmented) and end < len(hand_augmented):
            # 只绘制非零点之间的连接
            if np.any(hand_augmented[start] != 0) and np.any(hand_augmented[end] != 0):
                ax2.plot([hand_augmented[start, 0], hand_augmented[end, 0]],
                        [hand_augmented[start, 1], hand_augmented[end, 1]],
                        [hand_augmented[start, 2], hand_augmented[end, 2]],
                        'orange', alpha=0.7, linewidth=2)

    ax2.set_title('Augmented Data - 3D Keypoints with Connections', fontweight='bold', fontsize=14)
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_zlabel('Z Coordinate')
    ax2.set_xlim(axis_limits[0])
    ax2.set_ylim(axis_limits[1])
    ax2.set_zlim(axis_limits[2])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 设置更好的视角
    for ax in [ax1, ax2]:
        ax.view_init(elev=25, azim=135)  # 更好的视角角度

        # 设置等比例缩放
        ax.set_box_aspect([1,1,1])  # 使坐标轴比例相等

    plt.tight_layout()
    return fig

def visualize_sample_gestures(num_samples=5):
    """可视化不同类别的样本手势"""
    X, y = load_data('hand_gesture_data.npz')
    if X is None:
        return

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Gesture Samples Visualization by Category', fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for i, (class_id, gesture_name) in enumerate(GESTURES.items()):
        if i >= 10:  # 只显示前10个类别
            break

        # 找到该类别的样本
        class_indices = np.where(y == class_id)[0]
        if len(class_indices) == 0:
            continue

        # 随机选择一个样本
        sample_idx = np.random.choice(class_indices)
        sample = X[sample_idx]

        # 如果是序列数据，取中间帧
        if len(sample.shape) == 2:
            frame = sample[len(sample)//2]
        else:
            frame = sample

        # 分离姿势和手势关键点
        pose_points = frame[:99].reshape(-1, 3)
        hand_points = frame[99:].reshape(-1, 3)

        # 绘制2D投影
        axes[i].scatter(pose_points[:, 0], pose_points[:, 1], c='blue', alpha=0.6, s=15, label='Pose')
        axes[i].scatter(hand_points[:, 0], hand_points[:, 1], c='red', alpha=0.8, s=20, label='Hand')
        axes[i].set_title(f'{gesture_name}', fontweight='bold')
        axes[i].set_xlim(-0.2, 1.2)
        axes[i].set_ylim(-0.2, 1.2)
        axes[i].axis('off')

        if i == 0:  # 只在第一个子图显示图例
            axes[i].legend(loc='upper right')

    plt.tight_layout()
    return fig

def main():
    """主函数：生成所有可视化图表"""
    print("🎨 Starting dataset visualization chart generation...")

    # 1. Original dataset statistics
    print("\n📊 Analyzing original dataset...")
    X_original, y_original = load_data('hand_gesture_data.npz')
    if X_original is not None:
        fig1 = plot_dataset_statistics(X_original, y_original, "Original Dataset Statistics Analysis")
        fig1.savefig('original_dataset_analysis.png', dpi=150, bbox_inches='tight')
        print("✅ Original dataset statistics saved: original_dataset_analysis.png")

    # 2. Augmented dataset statistics
    print("\n📈 Analyzing augmented dataset...")
    X_augmented, y_augmented = load_data('hand_gesture_data_augmented.npz')
    if X_augmented is not None:
        fig2 = plot_dataset_statistics(X_augmented, y_augmented, "Augmented Dataset Statistics Analysis")
        fig2.savefig('augmented_dataset_analysis.png', dpi=150, bbox_inches='tight')
        print("✅ Augmented dataset statistics saved: augmented_dataset_analysis.png")

    # 3. Augmentation comparison
    print("\n🔄 Generating before vs after augmentation comparison...")
    fig3 = plot_augmentation_comparison()
    fig3.savefig('augmentation_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Augmentation comparison saved: augmentation_comparison.png")

    # 4. Gesture sample visualization
    print("\n👋 Generating gesture sample visualization...")
    fig4 = visualize_sample_gestures()
    fig4.savefig('gesture_samples_visualization.png', dpi=150, bbox_inches='tight')
    print("✅ Gesture sample visualization saved: gesture_samples_visualization.png")

    # 5. 3D keypoints comparison visualization
    print("\n🔲 Generating 3D keypoints comparison visualization...")
    fig5 = visualize_3d_keypoints_comparison()
    fig5.savefig('3d_keypoints_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ 3D keypoints comparison saved: 3d_keypoints_comparison.png")

    print("\n🎉 All visualization charts generated successfully!")
    print("📁 Generated files:")
    print("   - original_dataset_analysis.png")
    print("   - augmented_dataset_analysis.png")
    print("   - augmentation_comparison.png")
    print("   - gesture_samples_visualization.png")
    print("   - 3d_keypoints_comparison.png")

if __name__ == "__main__":
    main()