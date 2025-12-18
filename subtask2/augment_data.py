import numpy as np
import random
from scipy.spatial.transform import Rotation

def apply_random_transform(keypoints, rotation_range=30, scale_range=(0.8, 1.2), translate_range=(-0.1, 0.1), noise_std=0.01):
    """
    对关键点应用随机变换
    keypoints: (21, 3) 单个帧的关键点
    """
    transformed = keypoints.copy()

    # 1. 随机旋转
    if rotation_range > 0:
        # 随机旋转角度（度数）
        rot_x = np.random.uniform(-rotation_range, rotation_range)
        rot_y = np.random.uniform(-rotation_range, rotation_range)
        rot_z = np.random.uniform(-rotation_range, rotation_range)

        # 创建旋转矩阵
        rotation = Rotation.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=True)
        rot_matrix = rotation.as_matrix()

        # 应用旋转（以手掌中心为原点）
        palm_center = transformed[0]  # 假设关键点0是手掌中心
        transformed = transformed - palm_center
        transformed = np.dot(transformed, rot_matrix.T)
        transformed = transformed + palm_center

    # 2. 随机缩放
    if scale_range != (1.0, 1.0):
        scale = np.random.uniform(scale_range[0], scale_range[1])
        palm_center = transformed[0]
        transformed = transformed - palm_center
        transformed = transformed * scale
        transformed = transformed + palm_center

    # 3. 随机平移
    if translate_range != (0, 0):
        translate_x = np.random.uniform(translate_range[0], translate_range[1])
        translate_y = np.random.uniform(translate_range[0], translate_range[1])
        translate_z = np.random.uniform(translate_range[0], translate_range[1])
        transformed[:, 0] += translate_x
        transformed[:, 1] += translate_y
        transformed[:, 2] += translate_z

    # 4. 添加小噪声
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, transformed.shape)
        transformed += noise

    return transformed

def augment_dataset(X, y, augment_factor=4):
    """
    数据增强
    X: (n_samples, seq_len, 126) 原始数据
    y: (n_samples,) 标签
    augment_factor: 每个样本增强多少倍
    """
    n_samples, seq_len, feature_dim = X.shape
    augmented_X = []
    augmented_y = []

    # 保留原始数据
    augmented_X.extend(X)
    augmented_y.extend(y)

    print(f"原始数据: {n_samples} 个样本")

    # 为每个样本生成增强版本
    for i in range(n_samples):
        sample = X[i]  # (seq_len, 126)
        label = y[i]

        for aug_idx in range(augment_factor):
            # 对整个序列应用相同的变换参数
            augmented_sample = []

            # 为这个增强样本生成变换参数
            rot_x = np.random.uniform(-20, 20)
            rot_y = np.random.uniform(-20, 20)
            rot_z = np.random.uniform(-20, 20)
            scale = np.random.uniform(0.9, 1.1)
            translate_x = np.random.uniform(-0.05, 0.05)
            translate_y = np.random.uniform(-0.05, 0.05)
            translate_z = np.random.uniform(-0.05, 0.05)
            noise_std = np.random.uniform(0, 0.005)
            mirror = np.random.choice([True, False])  # 随机镜面

            for frame_idx in range(seq_len):
                frame_keypoints = sample[frame_idx].reshape(42, 3)  # 双手42点

                # 应用变换（对每手分别变换）
                if mirror:
                    # 镜面：交换左右手，并翻转x
                    transformed_left = apply_random_transform_single(
                        frame_keypoints[21:],  # 原来右手变左手
                        rot_x=rot_x, rot_y=rot_y, rot_z=rot_z,
                        scale=scale,
                        translate_x=translate_x, translate_y=translate_y, translate_z=translate_z,
                        noise_std=noise_std, mirror=True
                    )
                    transformed_right = apply_random_transform_single(
                        frame_keypoints[:21],  # 原来左手变右手
                        rot_x=rot_x, rot_y=rot_y, rot_z=rot_z,
                        scale=scale,
                        translate_x=translate_x, translate_y=translate_y, translate_z=translate_z,
                        noise_std=noise_std, mirror=True
                    )
                else:
                    transformed_left = apply_random_transform_single(
                        frame_keypoints[:21],  # 左/右手
                        rot_x=rot_x, rot_y=rot_y, rot_z=rot_z,
                        scale=scale,
                        translate_x=translate_x, translate_y=translate_y, translate_z=translate_z,
                        noise_std=noise_std, mirror=False
                    )
                    transformed_right = apply_random_transform_single(
                        frame_keypoints[21:],  # 另一手
                        rot_x=rot_x, rot_y=rot_y, rot_z=rot_z,
                        scale=scale,
                        translate_x=translate_x, translate_y=translate_y, translate_z=translate_z,
                        noise_std=noise_std, mirror=False
                    )
                transformed = np.concatenate([transformed_left, transformed_right])

                augmented_sample.append(transformed.reshape(126))

            augmented_X.append(np.array(augmented_sample))
            augmented_y.append(label)

        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{n_samples} 个样本")

    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)

    print(f"增强后数据: {len(augmented_X)} 个样本")
    print(f"标签分布: {np.bincount(augmented_y)}")

    return augmented_X, augmented_y

def apply_random_transform_single(keypoints, rot_x=0, rot_y=0, rot_z=0, scale=1.0,
                                translate_x=0, translate_y=0, translate_z=0, noise_std=0, mirror=False):
    """对单个帧应用指定的变换"""
    transformed = keypoints.copy()

    # 旋转
    if any([rot_x, rot_y, rot_z]):
        rotation = Rotation.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=True)
        rot_matrix = rotation.as_matrix()

        palm_center = transformed[0]
        transformed = transformed - palm_center
        transformed = np.dot(transformed, rot_matrix.T)
        transformed = transformed + palm_center

    # 缩放
    if scale != 1.0:
        palm_center = transformed[0]
        transformed = transformed - palm_center
        transformed = transformed * scale
        transformed = transformed + palm_center

    # 平移
    if any([translate_x, translate_y, translate_z]):
        transformed[:, 0] += translate_x
        transformed[:, 1] += translate_y
        transformed[:, 2] += translate_z

    # 镜面（左右翻转x坐标）
    if mirror:
        transformed[:, 0] = -transformed[:, 0]  # 翻转x

    # 噪声
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, transformed.shape)
        transformed += noise

    return transformed

if __name__ == "__main__":
    # 设置随机种子以便重现
    np.random.seed(42)
    random.seed(42)

    # 加载原始数据
    data = np.load('hand_gesture_data.npz')
    X = data['X']
    y = data['y']

    print("开始数据增强...")
    # 每个样本生成39个增强版本，总共40倍数据
    augmented_X, augmented_y = augment_dataset(X, y, augment_factor=39)

    # 保存增强后的数据
    np.savez('hand_gesture_data_augmented.npz', X=augmented_X, y=augmented_y)
    print(f"增强数据已保存到 hand_gesture_data_augmented.npz")
    print(f"最终数据集形状: {augmented_X.shape}")