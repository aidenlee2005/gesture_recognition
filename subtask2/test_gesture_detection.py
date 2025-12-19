#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进的手势检测逻辑
"""

import cv2
import numpy as np

def test_gesture_detection_logic():
    """测试手势检测的逻辑"""

    print("🧪 测试改进的手势检测逻辑...")

    # 模拟不同情况的特征数据

    # 情况1: 双手不动 - 只有身体姿势，没有手势
    keypoints_no_hands = np.zeros(225)  # 33*3 + 42*3
    # 填充一些身体姿势数据（前99个值）
    keypoints_no_hands[:99] = np.random.rand(99) * 0.1  # 小的随机值模拟身体姿势

    # 情况2: 有手势 - 身体姿势 + 手势数据
    keypoints_with_hands = keypoints_no_hands.copy()
    # 填充手势数据（后126个值，从索引99开始）
    keypoints_with_hands[99:] = np.random.rand(126) * 0.1

    # 测试函数
    def check_gesture_detection(keypoints_sequence, threshold=0.3):
        """检查手势检测逻辑"""
        sequence_array = np.array(keypoints_sequence)
        non_zero_ratio = np.mean(sequence_array != 0)

        # 检查最近几帧是否有手势检测
        recent_frames = keypoints_sequence[-5:] if len(keypoints_sequence) >= 5 else keypoints_sequence
        hand_detection_ratio = sum(1 for frame_keypoints in recent_frames
                                 if np.mean(frame_keypoints[99:]) != 0) / len(recent_frames)

        gesture_detected = hand_detection_ratio > threshold

        return {
            'non_zero_ratio': non_zero_ratio,
            'hand_detection_ratio': hand_detection_ratio,
            'gesture_detected': gesture_detected,
            'should_predict': non_zero_ratio > 0.3 and gesture_detected
        }

    # 测试情况1: 双手不动
    print("\n1. 测试双手不动的情况:")
    sequence_no_hands = [keypoints_no_hands] * 30
    result1 = check_gesture_detection(sequence_no_hands)
    print(f"   非零比例: {result1['non_zero_ratio']:.3f}")
    print(f"   手势检测比例: {result1['hand_detection_ratio']:.3f}")
    print(f"   检测到手势: {result1['gesture_detected']}")
    print(f"   应该预测: {result1['should_predict']}")

    # 测试情况2: 有手势
    print("\n2. 测试有手势的情况:")
    sequence_with_hands = [keypoints_with_hands] * 30
    result2 = check_gesture_detection(sequence_with_hands)
    print(f"   非零比例: {result2['non_zero_ratio']:.3f}")
    print(f"   手势检测比例: {result2['hand_detection_ratio']:.3f}")
    print(f"   检测到手势: {result2['gesture_detected']}")
    print(f"   应该预测: {result2['should_predict']}")

    # 测试情况3: 混合情况（大部分帧没有手势）
    print("\n3. 测试混合情况（大部分帧没有手势）:")
    mixed_sequence = [keypoints_no_hands] * 28 + [keypoints_with_hands] * 2
    result3 = check_gesture_detection(mixed_sequence)
    print(f"   非零比例: {result3['non_zero_ratio']:.3f}")
    print(f"   手势检测比例: {result3['hand_detection_ratio']:.3f}")
    print(f"   检测到手势: {result3['gesture_detected']}")
    print(f"   应该预测: {result3['should_predict']}")

    print("\n✅ 测试完成！")
    print("\n📊 预期结果:")
    print("   情况1 (双手不动): 不应该预测 -> 显示 'No Gesture Detected'")
    print("   情况2 (有手势): 应该预测 -> 显示具体手势")
    print("   情况3 (混合): 不应该预测 -> 显示 'No Gesture Detected'")

if __name__ == "__main__":
    test_gesture_detection_logic()