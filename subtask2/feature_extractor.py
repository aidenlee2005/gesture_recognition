# 改进的特征提取器，支持不同MediaPipe版本

import cv2
import numpy as np

class FeatureExtractor:
    """统一的特征提取器，支持新旧MediaPipe API"""

    def __init__(self):
        self.mp_version = None
        self.detector_type = None
        self._init_mediapipe()

    def _init_mediapipe(self):
        """初始化MediaPipe检测器"""
        try:
            import mediapipe as mp

            # 优先使用旧版API（更稳定），新版API作为备选
            if hasattr(mp, 'solutions'):
                # 旧版API可用
                self.mp_version = 'old'
                self._init_old_api()
            elif hasattr(mp, 'tasks') and mp.__version__ >= '0.10':
                # 新版API
                self.mp_version = 'new'
                self._init_new_api()
            else:
                raise ImportError(f"不支持的MediaPipe版本: {mp.__version__}")

        except ImportError:
            raise ImportError("MediaPipe未安装")

    def _init_old_api(self):
        """初始化旧版MediaPipe API"""
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5
        )
        self.detector_type = 'old'

    def _init_new_api(self):
        """初始化新版MediaPipe API"""
        print("⚠️  新版MediaPipe API需要下载模型文件，暂时使用简化实现")
        print("   建议使用 MediaPipe 0.9.x 或更早版本以获得最佳兼容性")

        # 对于新版API，我们暂时抛出异常建议使用旧版
        raise ImportError("新版MediaPipe API需要额外配置，请使用 MediaPipe <= 0.9.x")

    def extract_features(self, image):
        """提取特征，支持新旧API"""
        if self.detector_type == 'old':
            return self._extract_old_api(image)
        else:
            return self._extract_new_api(image)

    def _extract_old_api(self, image):
        """使用旧版API提取特征"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理姿势
        pose_results = self.pose_detector.process(image_rgb)
        pose_keypoints = []
        if pose_results.pose_landmarks:
            for lm in pose_results.pose_landmarks.landmark:
                pose_keypoints.extend([lm.x, lm.y, lm.z])
        else:
            pose_keypoints = [0] * (33 * 3)

        # 处理双手
        hand_results = self.hands_detector.process(image_rgb)
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

    def _extract_new_api(self, image):
        """新版API特征提取（未实现）"""
        raise NotImplementedError("新版MediaPipe API暂未实现")

    def close(self):
        """关闭检测器"""
        if self.detector_type == 'old':
            self.hands_detector.close()
            self.pose_detector.close()