import cv2
import mediapipe as mp
import torch
import numpy as np
import json
from model import GestureGRU

# 加载模型
model = GestureGRU()
model.load_state_dict(torch.load('gesture_gru.pth'))
model.eval()

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
            X = torch.tensor([sequence], dtype=torch.float32)
            with torch.no_grad():
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                gesture = GESTURES[predicted.item()]
                cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference()