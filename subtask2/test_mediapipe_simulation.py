#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试实际的MediaPipe检测结果模拟
"""

import cv2
import numpy as np
from feature_extractor import FeatureExtractor

def simulate_mediapipe_results():
    """模拟MediaPipe的检测结果"""

    print("🔬 模拟MediaPipe检测结果...")

    # 创建模拟的检测结果类
    class MockPoseResults:
        def __init__(self, has_pose=True):
            self.pose_landmarks = None
            if has_pose:
                # 模拟33个身体关键点
                self.pose_landmarks = MockLandmarks(33)

    class MockHandResults:
        def __init__(self, num_hands=0):
            self.multi_hand_landmarks = None
            if num_hands > 0:
                self.multi_hand_landmarks = []
                for _ in range(num_hands):
                    self.multi_hand_landmarks.append(MockLandmarks(21))

    class MockLandmarks:
        def __init__(self, num_points):
            self.landmark = []
            for i in range(num_points):
                landmark = MockLandmark()
                landmark.x = np.random.rand() * 0.2 + 0.4  # 0.4-0.6范围
                landmark.y = np.random.rand() * 0.2 + 0.3  # 0.3-0.5范围
                landmark.z = np.random.rand() * 0.1 - 0.05  # -0.05-0.05范围
                self.landmark.append(landmark)

    class MockLandmark:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    # 测试不同的场景
    scenarios = [
        ("双手不动", MockPoseResults(True), MockHandResults(0)),
        ("单手手势", MockPoseResults(True), MockHandResults(1)),
        ("双手手势", MockPoseResults(True), MockHandResults(2)),
        ("无姿势无手势", MockPoseResults(False), MockHandResults(0)),
    ]

    try:
        extractor = FeatureExtractor()

        for scenario_name, pose_results, hand_results in scenarios:
            print(f"\n📋 测试场景: {scenario_name}")

            # 创建一个测试帧
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # 模拟特征提取
            keypoints, _, _ = extractor._extract_old_api(frame)

            # 检查手势检测逻辑
            has_hands = hand_results.multi_hand_landmarks is not None and len(hand_results.multi_hand_landmarks) > 0
            has_pose = pose_results.pose_landmarks is not None

            # 计算手势检测比例（模拟最近5帧）
            hand_detection_ratio = 1.0 if has_hands else 0.0
            gesture_detected = hand_detection_ratio > 0.3

            print(f"   检测到姿势: {has_pose}")
            print(f"   检测到手势: {has_hands}")
            print(f"   手势检测比例: {hand_detection_ratio:.3f}")
            print(f"   判断为手势: {gesture_detected}")

            # 检查关键点数据
            pose_keypoints = keypoints[:99]  # 前99个是姿势
            hand_keypoints = keypoints[99:]  # 后126个是手势

            pose_nonzero = np.mean(pose_keypoints != 0)
            hand_nonzero = np.mean(hand_keypoints != 0)

            print(f"   姿势关键点非零比例: {pose_nonzero:.3f}")
            print(f"   手势关键点非零比例: {hand_nonzero:.3f}")

        extractor.close()

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    print("\n✅ 模拟测试完成！")

if __name__ == "__main__":
    simulate_mediapipe_results()