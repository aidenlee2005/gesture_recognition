# 改进的推理脚本 - 环境自适应版本

import cv2
import torch
import numpy as np
import json
from collections import Counter
from model import GestureGRU
from sklearn.preprocessing import StandardScaler
from feature_extractor import FeatureExtractor

def adaptive_inference():
    """自适应推理，根据环境调整参数"""

    print("🎯 启动环境自适应手势识别...")

    # 初始化组件
    try:
        extractor = FeatureExtractor()
        print("✅ 特征提取器初始化成功")
    except Exception as e:
        print(f"❌ 特征提取器初始化失败: {e}")
        return

    # 加载模型
    try:
        model = GestureGRU()
        model.load_state_dict(torch.load('gesture_gru_cv.pth'))
        model.eval()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 加载标准化器
    try:
        train_data = np.load('hand_gesture_data.npz')
        X_train = train_data['X']
        original_shape = X_train.shape
        X_reshaped = X_train.reshape(-1, original_shape[-1])
        scaler = StandardScaler()
        scaler.fit(X_reshaped)
        print("✅ 标准化器加载成功")
    except Exception as e:
        print(f"❌ 标准化器加载失败: {e}")
        return

    # 加载类别
    try:
        with open('classes.json', 'r') as f:
            GESTURES = json.load(f)
            GESTURES = {int(k): v for k, v in GESTURES.items()}
        print("✅ 类别加载成功")
    except Exception as e:
        print(f"❌ 类别加载失败: {e}")
        return

    # 自适应参数
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    # 环境分析阶段
    print("\n📊 环境分析中...")
    env_stats = analyze_environment(cap, extractor)
    adaptive_params = get_adaptive_parameters(env_stats)

    print("\n🎛️  自适应参数:")
    print(f"   置信度阈值: {adaptive_params['confidence_threshold']}")
    print(f"   缓冲区大小: {adaptive_params['buffer_size']}")
    print(f"   检测质量阈值: {adaptive_params['quality_threshold']}")

    # 推理阶段
    sequence = []
    seq_len = 30
    predictions = []
    confidences = []

    print("\n🚀 开始推理 (按'q'退出)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 提取特征
        try:
            keypoints, pose_results, hand_results = extractor.extract_features(frame)
        except Exception as e:
            print(f"特征提取错误: {e}")
            continue

        # 绘制检测结果
        draw_detection_results(frame, pose_results, hand_results)

        sequence.append(keypoints)

        if len(sequence) > seq_len:
            sequence.pop(0)

        if len(sequence) == seq_len:
            # 质量检查
            sequence_array = np.array(sequence)
            non_zero_ratio = np.mean(sequence_array != 0)

            if non_zero_ratio > adaptive_params['quality_threshold']:
                # 标准化和预测
                seq_reshaped = sequence_array.reshape(-1, sequence_array.shape[-1])
                seq_normalized = scaler.transform(seq_reshaped)
                seq_normalized = seq_normalized.reshape(sequence_array.shape)

                X = torch.tensor(seq_normalized, dtype=torch.float32).unsqueeze(0)

                try:
                    with torch.no_grad():
                        outputs = model(X)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)

                        conf_value = confidence.item()

                        if conf_value > adaptive_params['confidence_threshold']:
                            predictions.append(predicted.item())
                            confidences.append(conf_value)

                            if len(predictions) > adaptive_params['buffer_size']:
                                predictions.pop(0)
                                confidences.pop(0)

                        # 显示结果
                        display_results(frame, predictions, confidences, GESTURES, adaptive_params)

                except Exception as e:
                    print(f"预测错误: {e}")
                    cv2.putText(frame, f'Error: {str(e)[:30]}', (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f'Low Quality: {non_zero_ratio:.2f}',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Adaptive Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

def analyze_environment(cap, extractor, sample_frames=100):
    """分析环境条件"""
    print(f"   采样 {sample_frames} 帧进行环境分析...")

    brightness_values = []
    contrast_values = []
    detection_rates = []

    for i in range(sample_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # 分析图像质量
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(np.mean(gray))
        contrast_values.append(np.std(gray))

        # 分析检测质量
        try:
            keypoints, _, _ = extractor.extract_features(frame)
            detection_rate = np.mean(keypoints != 0)
            detection_rates.append(detection_rate)
        except:
            detection_rates.append(0)

    stats = {
        'avg_brightness': np.mean(brightness_values),
        'avg_contrast': np.mean(contrast_values),
        'avg_detection_rate': np.mean(detection_rates),
        'brightness_std': np.std(brightness_values),
        'detection_stability': 1 - np.std(detection_rates)  # 稳定性指标
    }

    print("\n   环境统计:")
    print(f"     平均亮度: {stats['avg_brightness']:.1f}")
    print(f"     平均对比度: {stats['avg_contrast']:.1f}")
    print(f"     检测率: {stats['avg_detection_rate']:.1f}")
    print(f"     亮度稳定性: {stats['brightness_std']:.1f}")
    print(f"     检测稳定性: {stats['detection_stability']:.1f}")
    return stats

def get_adaptive_parameters(env_stats):
    """根据环境统计调整参数"""

    # 基础参数
    params = {
        'confidence_threshold': 0.7,
        'buffer_size': 8,
        'quality_threshold': 0.3
    }

    # 根据检测质量调整
    detection_rate = env_stats['avg_detection_rate']
    if detection_rate > 0.7:
        params['confidence_threshold'] = 0.6  # 检测好时降低阈值
        params['quality_threshold'] = 0.2
    elif detection_rate > 0.5:
        params['confidence_threshold'] = 0.75  # 检测一般时提高阈值
        params['quality_threshold'] = 0.35
    else:
        params['confidence_threshold'] = 0.8  # 检测差时提高阈值
        params['quality_threshold'] = 0.4

    # 根据亮度调整
    brightness = env_stats['avg_brightness']
    if brightness < 80:
        params['buffer_size'] = 12  # 暗环境使用更大缓冲区
    elif brightness > 200:
        params['buffer_size'] = 6   # 亮环境使用小缓冲区

    return params

def draw_detection_results(frame, pose_results, hand_results):
    """绘制检测结果"""
    try:
        import mediapipe as mp

        # 根据API版本绘制
        if hasattr(mp, 'solutions'):
            # 旧版API
            mp_drawing = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands
            mp_pose = mp.solutions.pose

            if pose_results and pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if hand_results and hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            # 新版API - 简化绘制
            cv2.putText(frame, 'Detection active', (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    except Exception as e:
        cv2.putText(frame, f'Draw error: {str(e)[:20]}', (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def display_results(frame, predictions, confidences, GESTURES, params):
    """显示预测结果"""
    if len(predictions) >= 3:
        most_common = Counter(predictions).most_common(1)[0][0]
        avg_confidence = np.mean(confidences[-len(predictions):])
        gesture = GESTURES[most_common]

        # 根据置信度选择颜色
        if avg_confidence > 0.8:
            color = (0, 255, 0)  # 绿色 - 高置信度
        elif avg_confidence > 0.6:
            color = (0, 255, 255)  # 黄色 - 中等置信度
        else:
            color = (0, 165, 255)  # 橙色 - 低置信度

        cv2.putText(frame, f'Gesture: {gesture}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f'Confidence: {avg_confidence:.2f}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    elif len(predictions) > 0:
        gesture = GESTURES[predictions[-1]]
        cv2.putText(frame, f'Detecting: {gesture}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    else:
        cv2.putText(frame, 'Analyzing...', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 显示环境参数
    cv2.putText(frame, f'Threshold: {params["confidence_threshold"]:.2f}', (10, frame.shape[0] - 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f'Buffer: {params["buffer_size"]}', (10, frame.shape[0] - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f'Quality: {params["quality_threshold"]:.2f}', (10, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

if __name__ == "__main__":
    adaptive_inference()