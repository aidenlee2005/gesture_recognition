import cv2
import mediapipe as mp
import numpy as np
import json
import os

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 手势类别映射
GESTURES = {
    0: "One-hand OK",
    1: "One-hand Thumbs Up",
    2: "One-hand Yeah",
    3: "One-hand Fist",
    4: "One-hand Palm",
    5: "Two-hand OK",
    6: "Two-hand Thumbs Up",
    7: "Two-hand Yeah",
    8: "Two-hand Fist",
    9: "Two-hand Palm",
    10: "Two-hand House",
    11: "Two-hand Apart",
    12: "Two-hand Cross",
    13: "Two-hand Clap",
    14: "Two-hand Heart",
    15: "Two-hand V",
    16: "Two-hand Pray",
    17: "Two-hand Fist Bump",
    18: "Two-hand Open",
    19: "Two-hand Rock",
}

def extract_hand_landmarks(image):
    """从图像中提取手部关键点（支持双手）"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)
    keypoints = np.zeros(126)  # 两手63*2
    landmarks_list = []
    if results.multi_hand_landmarks:
        for i, landmarks in enumerate(results.multi_hand_landmarks[:2]):  # 最多两手
            offset = i * 63
            for lm in landmarks.landmark:
                keypoints[offset:offset+3] = [lm.x, lm.y, lm.z]
                offset += 3
            landmarks_list.append(landmarks)
    return keypoints, landmarks_list

def collect_data():
    """采集数据"""
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 加载现有数据
    data = []
    labels = []
    if os.path.exists('hand_gesture_data.npz'):
        existing = np.load('hand_gesture_data.npz')
        data = existing['X'].tolist()
        labels = existing['y'].tolist()
        print(f"加载现有数据: {len(data)} 样本")

    seq_len = 30  # 每段序列长度
    current_gesture = None
    sequence = []

    print("输入手势编号选择手势：0-19（0-4单手, 5-19双手）, q=退出")
    print("输入 's' 开始录制一段序列")

    while True:
        command = input("请输入命令: ").strip().lower()
        
        if command == 'q':
            break
        elif command in [str(i) for i in range(20)]:
            current_gesture = int(command)
            print(f"选择手势: {GESTURES[current_gesture]}")
        elif command == 's' and current_gesture is not None:
            if not cap.isOpened():
                print("摄像头未打开，无法录制")
                continue
            print("开始录制...")
            sequence = []
            for i in range(seq_len):
                ret, frame = cap.read()
                if not ret:
                    print(f"读取帧失败 at {i}")
                    break
                keypoints, landmarks = extract_hand_landmarks(frame)
                sequence.append(keypoints)  # 总是126维
                print(f"帧 {i+1}/{seq_len} 录制中...")
            if len(sequence) == seq_len:
                data.append(sequence)
                labels.append(current_gesture)
                print(f"录制完成，手势: {GESTURES[current_gesture]}，总样本: {len(data)}")
            else:
                print(f"录制失败，只录制了 {len(sequence)} 帧")
        else:
            print("无效命令，请输入 0-3 选择手势，或 's' 录制，或 'q' 退出")

    cap.release()

    # 保存数据
    if data:
        data = np.array(data)
        labels = np.array(labels)
        np.savez('hand_gesture_data.npz', X=data, y=labels)
        with open('classes.json', 'w') as f:
            json.dump(GESTURES, f)
        print(f"数据保存完成，总样本: {len(data)}")
    else:
        print("没有采集到数据")

if __name__ == "__main__":
    collect_data()