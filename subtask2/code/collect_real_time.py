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

# 加载类别
with open('classes.json', 'r') as f:
    GESTURES = json.load(f)

def extract_full_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 处理姿势
    pose_results = pose_detector.process(image_rgb)
    pose_keypoints = []
    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:
            pose_keypoints.extend([lm.x, lm.y, lm.z])
    else:
        pose_keypoints = [0] * (33 * 3)

    # 处理双手
    hand_results = hands_detector.process(image_rgb)
    hand_keypoints = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks[:2]:
            for lm in hand_landmarks.landmark:
                hand_keypoints.extend([lm.x, lm.y, lm.z])
        while len(hand_keypoints) < 42 * 3:
            hand_keypoints.extend([0, 0, 0])
    else:
        hand_keypoints = [0] * (42 * 3)

    all_keypoints = pose_keypoints + hand_keypoints
    return np.array(all_keypoints), pose_results, hand_results

def collect_real_time():
    """实时数据收集 - 按键录制"""
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 检查是否已有数据
    data_file = 'hand_gesture_data.npz'
    if os.path.exists(data_file):
        data = np.load(data_file)
        existing_X = data['X'].tolist()
        existing_y = data['y'].tolist()
        print(f"发现现有数据: {len(existing_X)} 个样本")
    else:
        existing_X = []
        existing_y = []

    print("\n=== 实时手势数据收集 ===")
    print("操作说明:")
    print("  数字键 0-9: 选择要录制的手势")
    print("  空格键: 开始录制当前手势 (30帧)")
    print("  'd'键: 删除上一条记录")
    print("  'q': 退出程序")
    print("\n手势列表:")
    for i, gesture in GESTURES.items():
        print(f"  {i}: {gesture}")

    current_gesture = None
    sequence = []
    seq_len = 30
    is_recording = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, pose_results, hand_results = extract_full_landmarks(frame)

        # 绘制姿势和双手
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 显示状态信息
        if is_recording:
            progress = len(sequence) / seq_len * 100
            cv2.putText(frame, f'RECORDING: {GESTURES[str(current_gesture)]} ({len(sequence)}/{seq_len})',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'Progress: {progress:.1f}%', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif current_gesture is not None:
            cv2.putText(frame, f'Selected: {GESTURES[str(current_gesture)]} - Press SPACE to record',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(frame, 'Press 0-9 to select gesture, SPACE to record, D to delete last',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.putText(frame, f'Total samples: {len(existing_X)}', (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Gesture Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d'):  # 删除上一条记录
            if existing_X:
                removed_sequence = existing_X.pop()
                removed_label = existing_y.pop()
                print(f"删除了上一条记录：手势 {removed_label} ({GESTURES[str(removed_label)]})，总样本: {len(existing_X)}")
            else:
                print("没有记录可删除")
        elif key >= ord('0') and key <= ord('9'):
            gesture_id = key - ord('0')
            if str(gesture_id) in GESTURES:
                current_gesture = gesture_id
                print(f"选择手势: {GESTURES[str(gesture_id)]}")
                is_recording = False
        elif key == ord(' ') and current_gesture is not None:  # 空格键开始录制
            if not is_recording:
                is_recording = True
                sequence = []
                print(f"开始录制手势: {GESTURES[str(current_gesture)]}")
        elif is_recording:
            # 检查数据质量
            non_zero_ratio = np.mean(keypoints != 0)
            if non_zero_ratio > 0.4:  # 质量阈值
                sequence.append(keypoints)
            else:
                # 质量差时显示警告
                cv2.putText(frame, 'QUALITY WARNING: Adjust position/lighting', (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if len(sequence) >= seq_len:
                # 录制完成
                existing_X.append(sequence)
                existing_y.append(current_gesture)
                print(f"录制完成! 手势: {GESTURES[str(current_gesture)]}, 总样本数: {len(existing_X)}")
                is_recording = False
                current_gesture = None
                sequence = []

    # 保存数据
    if existing_X:
        X = np.array(existing_X)
        y = np.array(existing_y)
        np.savez(data_file, X=X, y=y)
        print(f"\n数据已保存到 {data_file}")
        print(f"总样本数: {len(X)}")
        print(f"类别分布: {np.bincount(y)}")

        # 建议每个类别至少收集多少样本
        unique, counts = np.unique(y, return_counts=True)
        print("\n每个类别的样本数:")
        for gesture_id, count in zip(unique, counts):
            gesture_name = GESTURES[str(gesture_id)]
            status = "✓" if count >= 10 else "⚠️"
            print(f"  {gesture_id} ({gesture_name}): {count} {status}")

        min_samples = min(counts) if len(counts) > 0 else 0
        if min_samples < 10:
            print(f"\n⚠️  建议: 每个类别至少收集10个样本，最少的类别只有{min_samples}个")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_real_time()