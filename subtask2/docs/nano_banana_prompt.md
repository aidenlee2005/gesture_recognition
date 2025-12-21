# GRU手势识别系统Pipeline示意图 - Nano Banana Prompt

Create a clean, minimalist pipeline diagram showing the training and inference workflow of our GRU-based gesture recognition system. Use a horizontal layout with clear sections for training and inference phases. Style should be modern and professional with subtle colors.

## Overall Layout:
- **Left side**: Training Pipeline (Data → Model)
- **Right side**: Inference Pipeline (Input → Prediction)
- **Center**: GRU Model Architecture (detailed internal structure)
- Use connecting arrows to show data flow
- Include key parameters and dimensions

## Training Pipeline (Left Section):

### Data Collection & Processing:
1. **Raw Data Collection**
   - Input: Camera frames (640x480 RGB)
   - Process: MediaPipe pose/hand detection
   - Output: 225D keypoints (33 pose + 42 hand × 3 coords)

2. **Data Augmentation**
   - Input: Raw gesture sequences (30 frames × 225D)
   - Process: Rotation, scaling, noise addition
   - Output: Augmented dataset (4000 samples)

3. **Dataset Preparation**
   - Input: Augmented data
   - Process: Train/Val/Test split (80/10/10)
   - Output: Normalized sequences with StandardScaler

### Model Training:
4. **GRU Model Training**
   - Input: Training sequences (batch_size=32)
   - Process: Cross-validation, early stopping
   - Output: Trained model (gesture_gru_cv.pth)
   - Metrics: 99% training accuracy

## Inference Pipeline (Right Section):

### Real-time Processing:
1. **Feature Extraction**
   - Input: Live camera frames
   - Process: MediaPipe detection (pose + hands)
   - Output: 225D keypoints per frame

2. **Sequence Buffering**
   - Input: Individual frame features
   - Process: Sliding window (30 frames)
   - Output: Complete gesture sequence

3. **Quality Check**
   - Input: Feature sequence
   - Process: Non-zero ratio + hand detection validation
   - Output: Quality score (0-1)

4. **Standardization**
   - Input: Raw sequence
   - Process: StandardScaler transform
   - Output: Normalized sequence

5. **GRU Inference**
   - Input: Normalized sequence (1 × 30 × 225)
   - Process: Forward pass through trained model
   - Output: Class probabilities (10 classes)

6. **Result Processing**
   - Input: Raw predictions
   - Process: Confidence thresholding + buffer averaging
   - Output: Final gesture prediction

## GRU Model Architecture (Center Section - Detailed):

### Input Layer:
- **Dimensions**: (batch_size, seq_len=30, input_size=225)
- **Features**: 33 pose keypoints + 42 hand keypoints × 3 coordinates

### GRU Layers (3 layers):
- **GRU Layer 1**: input_size=225 → hidden_size=256
- **GRU Layer 2**: hidden_size=256 → hidden_size=256
- **GRU Layer 3**: hidden_size=256 → hidden_size=256
- **Bidirectional**: False (unidirectional)
- **Dropout**: 0.5 between layers

### Output Layer:
- **Input**: Final hidden state (256D)
- **Linear Layer**: 256 → 10 classes
- **Activation**: Softmax
- **Output**: Class probabilities

### Key Parameters:
- **Sequence Length**: 30 frames
- **Input Dimensions**: 225 (33×3 + 42×3)
- **Hidden Size**: 256
- **Num Layers**: 3
- **Dropout**: 0.5
- **Num Classes**: 10 gestures

## Visual Style Requirements:
- **Color Scheme**: 
  - Training: Blue tones (#1e40af to #3b82f6)
  - Inference: Green tones (#059669 to #10b981)
  - Model: Purple/Gray tones (#6b46c1 to #9ca3af)
- **Typography**: Clean sans-serif fonts
- **Icons**: Simple geometric shapes for components
- **Arrows**: Thin, directional arrows with labels
- **Spacing**: Generous white space between components
- **Labels**: Clear, concise text for each component
- **Dimensions**: Show tensor shapes and key parameters

## Additional Elements:
- **Environment Adaptation**: Show adaptive parameters (confidence threshold, buffer size, quality threshold)
- **Gesture Classes**: List the 10 gesture types (Hello, Thank you, Sorry, You, Goodbye, I, Love, Help, Eat, Drink)
- **Performance Metrics**: Show typical accuracy (99% training, 80-90% real-time)
- **Data Flow**: Use different arrow styles for training vs inference data

Make the diagram comprehensive yet easy to read, with all key components clearly labeled and connected.