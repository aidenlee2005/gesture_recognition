import numpy as np
import random
from scipy.spatial.transform import Rotation

def apply_temporal_augmentation(sequence, speed_factor_range=(0.8, 1.2), noise_factor=0.05):
    """
    对时间序列应用时间增强
    sequence: (seq_len, features)
    """
    seq_len, features = sequence.shape
    
    # 速度扰动
    speed_factor = np.random.uniform(speed_factor_range[0], speed_factor_range[1])
    new_seq_len = int(seq_len * speed_factor)
    
    if new_seq_len < seq_len:
        # 加速：插值
        indices = np.linspace(0, seq_len-1, new_seq_len)
        augmented = np.zeros((seq_len, features))
        for i in range(features):
            augmented[:, i] = np.interp(np.arange(seq_len), indices, sequence[:new_seq_len, i])
    else:
        # 减速：重复帧
        indices = np.linspace(0, seq_len-1, new_seq_len)
        temp_sequence = np.zeros((new_seq_len, features))
        for i in range(features):
            temp_sequence[:, i] = np.interp(np.arange(new_seq_len), np.arange(seq_len), sequence[:, i])
        augmented = temp_sequence[:seq_len]  # 截断到原长度
    
    # 添加时间噪声
    noise = np.random.normal(0, noise_factor, augmented.shape)
    augmented += noise
    
    return augmented

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
        body_keypoints[:, 0] += translate_x
        body_keypoints[:, 1] += translate_y
        body_keypoints[:, 2] += translate_z
        hand_keypoints[:, 0] += translate_x
        hand_keypoints[:, 1] += translate_y
        hand_keypoints[:, 2] += translate_z
    
    # 4. 添加小噪声
    if noise_std > 0:
        body_noise = np.random.normal(0, noise_std, body_keypoints.shape)
        hand_noise = np.random.normal(0, noise_std, hand_keypoints.shape)
        body_keypoints += body_noise
        hand_keypoints += hand_noise
    
    # 重新组合
    transformed[:33] = body_keypoints
    transformed[33:] = hand_keypoints
    
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
            # 随机选择增强类型
            aug_type = np.random.choice(['spatial', 'temporal', 'both'])
            
            if aug_type in ['spatial', 'both']:
                # 空间增强
                augmented_seq = []
                for frame in sample:
                    transformed_frame = apply_random_transform_full(frame.reshape(-1, 3), 
                                                                   rotation_range=20, 
                                                                   scale_range=(0.9, 1.1), 
                                                                   translate_range=(-0.05, 0.05), 
                                                                   noise_std=np.random.uniform(0, 0.005))
                    augmented_seq.append(transformed_frame.flatten())
                augmented_seq = np.array(augmented_seq)
            else:
                augmented_seq = sample.copy()
            
            if aug_type in ['temporal', 'both']:
                # 时间增强
                augmented_seq = apply_temporal_augmentation(augmented_seq, 
                                                          speed_factor_range=(0.8, 1.2), 
                                                          noise_factor=0.02)
            
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