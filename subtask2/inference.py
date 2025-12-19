import cv2
import mediapipe as mp
import torch
import numpy as np
import json
from collections import Counter
from model import GestureGRU
from sklearn.preprocessing import StandardScaler

# 加载模型
model = GestureGRU()
model.load_state_dict(torch.load('gesture_gru_cv.pth'))  # 使用交叉验证训练的模型
model.eval()

# 加载标准化器（使用训练时相同的数据）
train_data = np.load('hand_gesture_data.npz')  # 使用原始数据，不是增强数据
X_train = train_data['X']
original_shape = X_train.shape
X_reshaped = X_train.reshape(-1, original_shape[-1])
scaler = StandardScaler()
scaler.fit(X_reshaped)

# 加载类别
with open('classes.json', 'r') as f:
    GESTURES = json.load(f)
    GESTURES = {int(k): v for k, v in GESTURES.items()}

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose_detector = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_full_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 处理姿势
    pose_results = pose_detector.process(image_rgb)
    pose_keypoints = []
    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:
            pose_keypoints.extend([lm.x, lm.y, lm.z])
    else:
        pose_keypoints = [0] * (33 * 3)  # 33点 * 3坐标
    
    # 处理双手
    hand_results = hands_detector.process(image_rgb)
    hand_keypoints = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks[:2]:  # 最多两只手
            for lm in hand_landmarks.landmark:
                hand_keypoints.extend([lm.x, lm.y, lm.z])
        # 如果少于两只手，填充零
        while len(hand_keypoints) < 42 * 3:
            hand_keypoints.extend([0, 0, 0])
    else:
        hand_keypoints = [0] * (42 * 3)  # 42点（21*2手）* 3坐标
    
    # 拼接所有关键点
    all_keypoints = pose_keypoints + hand_keypoints
    return np.array(all_keypoints), pose_results, hand_results

def inference():
    cap = cv2.VideoCapture(1)
    sequence = []
    seq_len = 30
    predictions = []  # 存储最近的预测结果
    confidences = []  # 存储对应的置信度
    buffer_size = 8  # 增加缓冲区大小，提高稳定性
    confidence_threshold = 0.7  # 设置置信度阈值

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, pose_results, hand_results = extract_full_landmarks(frame)
        
        # 绘制姿势
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # 绘制双手
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        sequence.append(keypoints)

        if len(sequence) > seq_len:
            sequence.pop(0)

        if len(sequence) == seq_len:
            # 检查序列质量（非零值比例）
            sequence_array = np.array(sequence)
            non_zero_ratio = np.mean(sequence_array != 0)
            
            if non_zero_ratio > 0.3:  # 如果检测质量足够好
                # 应用标准化
                seq_reshaped = sequence_array.reshape(-1, sequence_array.shape[-1])
                seq_normalized = scaler.transform(seq_reshaped)
                seq_normalized = seq_normalized.reshape(sequence_array.shape)
                
                X = torch.tensor(seq_normalized, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(X)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    conf_value = confidence.item()
                    
                    # 只有当置信度足够高时才添加到缓冲区
                    if conf_value > confidence_threshold:
                        predictions.append(predicted.item())
                        confidences.append(conf_value)
                        
                        if len(predictions) > buffer_size:
                            predictions.pop(0)
                            confidences.pop(0)
                    
                    # 显示预测结果
                    if len(predictions) >= 3:  # 至少需要3个预测
                        most_common = Counter(predictions).most_common(1)[0][0]
                        avg_confidence = np.mean(confidences[-len(predictions):])  # 最近预测的平均置信度
                        gesture = GESTURES[most_common]
                        cv2.putText(frame, f'Gesture: {gesture} (Conf: {avg_confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif len(predictions) > 0:
                        # 显示最新预测
                        gesture = GESTURES[predictions[-1]]
                        cv2.putText(frame, f'Gesture: {gesture} (Conf: {confidences[-1]:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, 'Detecting...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                # 检测质量差
                cv2.putText(frame, 'Poor detection - adjust position', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # 清空缓冲区
                predictions.clear()
                confidences.clear()

        cv2.imshow('Hand Gesture Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference()