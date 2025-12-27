# 手势识别系统

## 项目概述

本项目实现了一个基于RGB视频和关键点检测的轻量级手势识别系统，使用GRU时序建模，支持单手简单手势和双手复杂手语的实时识别。

项目分为两个子任务：
- **Subtask1**：识别5个基本单手手势（OK、Thumbs Up、Yeah、Fist、Palm）
- **Subtask2**：识别10个手语词汇（Hello、Thank you、Sorry、You、Goodbye、I、Love、Help、Eat、Drink）

## 主要特性

- **轻量级设计**：仅使用RGB视频，无需深度相机
- **实时性能**：在标准硬件上达到25 FPS
- **数据增强**：空间和时间增强提高鲁棒性
- **多模态输入**：Subtask2集成姿势和手部关键点
- **自适应推理**：动态阈值和缓冲区实现稳定预测

## 安装

### 前置条件
- Python 3.8+
- PyTorch 2.0+
- 摄像头访问权限

### 依赖包
```bash
pip install torch torchvision torchaudio
pip install mediapipe opencv-python numpy scikit-learn scipy matplotlib seaborn
```

## 快速开始

### 1. 数据收集
```bash
# Subtask1
cd subtask1/code
python collect_data.py

# Subtask2
cd ../../subtask2/code
python collect_real_time.py
```

### 2. 数据增强
```bash
python augment_data.py  # 生成增强数据集
```

### 3. 训练
```bash
# Subtask1
cd subtask1/code
python train.py

# Subtask2（推荐使用交叉验证）
cd ../../subtask2/code
python train_improved.py
```

### 4. 推理
```bash
# 基础推理
python inference.py

# 高级自适应推理（Subtask2）
python adaptive_inference.py
```

### 5. 评估
```bash
python test_acc.py  # 生成准确率报告和混淆矩阵
```

## 结果

- **Subtask2 测试性能**：
  - 准确率：84.33%
  - 加权F1分数：83%
  - 宏平均F1分数：82%

评估过程中会生成详细的分类报告和混淆矩阵。

## 项目结构

```
gesture/
├── subtask1/
│   ├── code/          # 源代码
│   ├── data/          # 数据集（.npz文件）
│   ├── models/        # 训练模型（.pth文件）
│   └── images/        # 生成的图表
├── subtask2/
│   ├── code/          # 源代码
│   ├── data/          # 数据集
│   ├── models/        # 训练模型
│   └── images/        # 生成的图表
└── README.md          # 本文件
```

## 使用注意事项

- 确保光照稳定以提高关键点检测效果
- 数据增强显著提升性能但会增加训练时间
- Subtask2模型复杂度更高，需要更多计算资源
- 在生产环境中使用自适应推理

## 贡献

本项目仅供学习使用。欢迎fork和修改。

## 许可证

MIT许可证