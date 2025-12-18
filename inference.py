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
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints), landmarks
    return None, None

def inference():
    cap = cv2.VideoCapture(0)
    sequence = []
    seq_len = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, landmarks = extract_hand_landmarks(frame)
        if keypoints is not None:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            sequence.append(keypoints)
        else:
            sequence.append(np.zeros(63))

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