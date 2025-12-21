# Subtask2 Code Files Documentation

This directory contains the core Python scripts for the gesture recognition project (Subtask 2: Full Pose + Hand).

## File Descriptions

### Core Model and Training
- **model.py**: Defines the GRU-based neural network model for gesture classification. Includes model architecture, forward pass, and initialization.
- **dataset.py**: Handles data loading, preprocessing, and dataset creation for training/validation. Supports sequence data with keypoints.
- **train.py**: Main training script. Loads data, trains the model, and saves checkpoints.
- **train_improved.py**: Enhanced training script with additional optimizations, such as improved data handling or hyperparameters.

### Data Processing
- **augment_data.py**: Performs data augmentation on gesture sequences to increase dataset diversity (e.g., rotations, noise).
- **collect_data.py**: Script for collecting gesture data from camera input using MediaPipe. Saves sequences to NPZ files.
- **collect_real_time.py**: Real-time data collection script for continuous gesture recording and processing.
- **feature_extractor.py**: Extracts and processes keypoints features from raw MediaPipe detections.

### Inference and Evaluation
- **inference.py**: Runs inference on trained models. Loads model, processes input sequences, and outputs predictions.
- **adaptive_inference.py**: Advanced inference with adaptive thresholding and confidence-based decision making.
- **visualize.py**: Generates visualizations for data analysis, model performance, and keypoints display (e.g., plots, 3D renders).

## Usage Notes
- Run training: `python train.py`
- Run inference: `python inference.py`
- Data collection: `python collect_data.py`
- Ensure dependencies are installed via `../docs/requirements.txt`

For more details, refer to the main project documentation in `../docs/`.