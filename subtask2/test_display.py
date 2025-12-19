#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试双手不动时的显示效果
"""

import cv2
import numpy as np
from adaptive_inference import display_results

def test_display_results():
    """测试display_results函数在不同情况下的表现"""

    # 创建一个测试帧
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 测试手势字典
    GESTURES = {
        0: "Hello",
        1: "Thank you",
        2: "Sorry",
        3: "You",
        4: "Goodbye",
        5: "I",
        6: "Love",
        7: "Help",
        8: "Eat",
        9: "Drink"
    }

    # 测试参数
    params = {
        'confidence_threshold': 0.7,
        'buffer_size': 8,
        'quality_threshold': 0.3
    }

    print("🧪 测试display_results函数...")

    # 测试1: 空预测列表（双手不动的情况）
    print("\n1. 测试空预测列表（双手不动）:")
    display_results(frame.copy(), [], [], GESTURES, params)
    cv2.imshow('Test 1: No Predictions', frame)
    cv2.waitKey(2000)

    # 测试2: 低置信度预测
    print("\n2. 测试低置信度预测:")
    predictions = [0, 0, 0]  # 都是Hello
    confidences = [0.5, 0.4, 0.6]  # 平均置信度低于阈值
    display_results(frame.copy(), predictions, confidences, GESTURES, params)
    cv2.imshow('Test 2: Low Confidence', frame)
    cv2.waitKey(2000)

    # 测试3: 高置信度预测
    print("\n3. 测试高置信度预测:")
    predictions = [7, 7, 7]  # 都是Help
    confidences = [0.9, 0.85, 0.95]  # 高置信度
    display_results(frame.copy(), predictions, confidences, GESTURES, params)
    cv2.imshow('Test 3: High Confidence', frame)
    cv2.waitKey(2000)

    # 测试4: 单个预测
    print("\n4. 测试单个预测:")
    predictions = [1]  # Thank you
    confidences = [0.8]  # 高置信度
    display_results(frame.copy(), predictions, confidences, GESTURES, params)
    cv2.imshow('Test 4: Single Prediction', frame)
    cv2.waitKey(2000)

    cv2.destroyAllWindows()
    print("\n✅ 测试完成！")

if __name__ == "__main__":
    test_display_results()