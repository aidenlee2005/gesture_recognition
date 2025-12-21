# Two-Hand Gesture Recognition with MediaPipe and GRU

这是一个基于 MediaPipe 和 GRU 网络的双双手势识别项目，使用 RGB 摄像头提取双手关键点（126维），进行实时手势分类。

## 项目结构

- `collect_data.py`: 数据采集脚本，从摄像头录制手势序列。
- `dataset.py`: PyTorch 数据集类。
- `model.py`: GRU 模型定义。
- `train.py`: 训练脚本。
- `inference.py`: 实时推理脚本。
- `visualize.py`: 数据可视化脚本。
- `requirements.txt`: 依赖包列表。
- `classes.json`: 手势类别映射。
- `hand_gesture_data.npz`: 采集的数据文件（运行后生成）。
- `gesture_gru.pth`: 训练后的模型文件（训练后生成）。

## 环境要求

- Python 3.8+
- macOS / Windows / Linux
- 摄像头（内置或外接）

## 安装

1. 克隆仓库：
   ```
   git clone https://github.com/aidenlee2005/gesture_recognition.git
   cd gesture_recognition
   ```

2. 创建虚拟环境（推荐）：
   ```
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # 或 venv\Scripts\activate  # Windows
   ```

3. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

## 使用指南

### 1. 数据采集

运行数据采集脚本：
```
python collect_data.py
```

- 窗口显示摄像头画面和手部骨架。
- 按 `0`、`1`、`2`、`3` 选择手势（OK, Thumbs Up, Peace, Fist）。
- 按 `s` 开始录制 30 帧序列。
- 按 `q` 退出并保存数据。

数据累积保存，每次运行追加新样本。

### 2. 训练模型

采集数据后，运行训练：
```
python train.py
```

- 自动划分训练/验证集。
- 训练 25 轮，保存最佳模型为 `gesture_gru.pth`。

### 3. 实时推理

训练后，运行推理：
```
python inference.py
```

- 加载模型，摄像头实时预测手势。
- 画面显示预测结果。

### 4. 数据可视化

查看采集的数据：
```
python visualize.py
```

- 显示第一个样本的第一帧 3D 骨架。

## 手势类别

- 0: OK
- 1: Thumbs Up
- 2: Peace
- 3: Fist

## 注意事项

- 确保摄像头权限（macOS 需要允许 Terminal 访问）。
- 数据量建议每类 20-50 样本。
- 如果 MediaPipe 导入错误，重新安装：`pip install mediapipe==0.10.5`。
- 训练时间取决于数据量，GPU 可加速。

## 许可证

MIT License