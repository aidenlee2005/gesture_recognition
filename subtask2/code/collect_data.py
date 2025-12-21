import cv2
import mediapipe as mp
import numpy as np
import json
import os

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose_detector = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 手势类别映射
GESTURES = {
    0: "Hello",
    1: "Thank you",
    2: "Sorry",
    3: "Please",
    4: "Goodbye",
    5: "I",
    6: "Love",
    7: "Help",
    8: "Eat",
    9: "Drink",
}

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

    print("输入手势编号选择手势：0-9（常用手语）, q=退出")
    print("手势列表：")
    for k, v in GESTURES.items():
        print(f"  {k}: {v}")
    print("输入 's' 开始录制一段序列，'d' 删除上一条记录")

    while True:
        command = input("请输入命令: ").strip().lower()
        
        if command == 'q':
            break
        elif command in [str(i) for i in range(10)]:
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
                keypoints, pose_results, hand_results = extract_full_landmarks(frame)
                sequence.append(keypoints)  # 225维
                # 绘制姿势和双手
                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.imshow('Recording', frame)
                cv2.waitKey(1)
                print(f"帧 {i+1}/{seq_len} 录制中...")
            cv2.destroyWindow('Recording')
            if len(sequence) == seq_len:
                data.append(sequence)
                labels.append(current_gesture)
                print(f"录制完成，手势: {GESTURES[current_gesture]}，总样本: {len(data)}")
            else:
                print(f"录制失败，只录制了 {len(sequence)} 帧")
        elif command == 'd':
            if data:
                removed_sequence = data.pop()
                removed_label = labels.pop()
                print(f"删除了上一条记录：手势 {removed_label} ({GESTURES[removed_label]})，总样本: {len(data)}")
            else:
                print("没有记录可删除")
        else:
            print("无效命令，请输入 0-9 选择手势，或 's' 录制，或 'd' 删除上一条，或 'q' 退出")

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