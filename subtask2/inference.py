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
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)
    keypoints = np.zeros(126)
    landmarks_list = []
    if results.multi_hand_landmarks:
        for i, landmarks in enumerate(results.multi_hand_landmarks[:2]):
            offset = i * 63
            for lm in landmarks.landmark:
                keypoints[offset:offset+3] = [lm.x, lm.y, lm.z]
                offset += 3
            landmarks_list.append(landmarks)
    return keypoints, landmarks_list

def inference():
    cap = cv2.VideoCapture(1)
    sequence = []
    seq_len = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, landmarks_list = extract_hand_landmarks(frame)
        for landmarks in landmarks_list:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
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