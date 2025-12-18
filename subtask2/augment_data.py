import numpy as np
import random
from scipy.spatial.transform import Rotation

def apply_random_transform_full(keypoints, rotation_range=30, scale_range=(0.8, 1.2), translate_range=(-0.1, 0.1), noise_std=0.01):
    """
    对完整关键点（身体+双手）应用随机变换
    keypoints: (75, 3) 单个帧的关键点 [33身体 + 42双手]
    """
    transformed = keypoints.copy()
    
    # 分离身体和双手
    body_keypoints = transformed[:33]  # 33点身体
    hand_keypoints = transformed[33:]  # 42点双手
    
    # 计算变换中心：优先使用躯干中心（肩+髋），否则使用肩膀中心
    shoulders = body_keypoints[[11, 12]]  # 左右肩
    hips = body_keypoints[[23, 24]]  # 左右髋
    
    # 检查髋部是否可用（不为0）
    hips_available = np.any(hips != 0)
    if hips_available:
        body_center = np.mean(np.vstack([shoulders, hips]), axis=0)  # 肩+髋平均
    else:
        body_center = np.mean(shoulders, axis=0)  # 仅肩膀平均
    
    # 1. 随机旋转
    if rotation_range > 0:
        rot_x = np.random.uniform(-rotation_range, rotation_range)
        rot_y = np.random.uniform(-rotation_range, rotation_range)
        rot_z = np.random.uniform(-rotation_range, rotation_range)
        
        rotation = Rotation.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=True)
        rot_matrix = rotation.as_matrix()
        
        # 以身体中心为原点旋转
        body_keypoints = body_keypoints - body_center
        body_keypoints = np.dot(body_keypoints, rot_matrix.T)
        body_keypoints = body_keypoints + body_center
        
        hand_keypoints = hand_keypoints - body_center
        hand_keypoints = np.dot(hand_keypoints, rot_matrix.T)
        hand_keypoints = hand_keypoints + body_center
    
    # 2. 随机缩放
    if scale_range != (1.0, 1.0):
        scale = np.random.uniform(scale_range[0], scale_range[1])
        body_keypoints = body_keypoints - body_center
        body_keypoints = body_keypoints * scale
        body_keypoints = body_keypoints + body_center
        
        hand_keypoints = hand_keypoints - body_center
        hand_keypoints = hand_keypoints * scale
        hand_keypoints = hand_keypoints + body_center
    
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
    数据增强（支持身体+双手关键点）
    X: (n_samples, seq_len, 225) 原始数据 [99身体 + 126双手]
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
        sample = X[i]  # (seq_len, 225)
        label = y[i]

        for aug_idx in range(augment_factor):
            augmented_sample = []

            # 生成变换参数
            rot_x = np.random.uniform(-20, 20)
            rot_y = np.random.uniform(-20, 20)
            rot_z = np.random.uniform(-20, 20)
            scale = np.random.uniform(0.9, 1.1)
            translate_x = np.random.uniform(-0.05, 0.05)
            translate_y = np.random.uniform(-0.05, 0.05)
            translate_z = np.random.uniform(-0.05, 0.05)
            noise_std = np.random.uniform(0, 0.005)
            
            # 时间增强参数
            time_warp = np.random.choice([True, False], p=[0.3, 0.7])  # 30%概率时间扭曲
            frame_drop = np.random.choice([True, False], p=[0.2, 0.8])  # 20%概率随机丢帧

            for frame_idx in range(seq_len):
                frame_keypoints = sample[frame_idx].reshape(75, 3)  # 33+42=75点

                # 应用空间变换
                transformed_frame = apply_random_transform_full(
                    frame_keypoints,
                    rotation_range=20,
                    scale_range=(0.9, 1.1),
                    translate_range=(-0.05, 0.05),
                    noise_std=noise_std
                )
                
                augmented_sample.append(transformed_frame.flatten())

            # 时间增强
            augmented_seq = np.array(augmented_sample)  # (seq_len, 225)
            
            if time_warp:
                # 简单时间扭曲：随机重采样
                indices = np.linspace(0, seq_len-1, seq_len)
                warp_factor = np.random.uniform(0.8, 1.2)
                warped_indices = indices * warp_factor
                warped_indices = np.clip(warped_indices, 0, seq_len-1).astype(int)
                augmented_seq = augmented_seq[warped_indices]
            
            if frame_drop and seq_len > 20:  # 确保不丢太多帧
                # 随机丢弃10%的帧
                drop_indices = np.random.choice(seq_len, size=int(seq_len * 0.1), replace=False)
                keep_indices = np.setdiff1d(np.arange(seq_len), drop_indices)
                augmented_seq = augmented_seq[keep_indices]
                # 插值回原长度（简单重复）
                if len(augmented_seq) < seq_len:
                    diff = seq_len - len(augmented_seq)
                    augmented_seq = np.concatenate([augmented_seq, augmented_seq[-diff:]])
            
            augmented_X.append(augmented_seq)
            augmented_y.append(label)
            
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)
    
    print(f"增强后数据: {len(augmented_X)} 个样本")
    return augmented_X, augmented_y

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