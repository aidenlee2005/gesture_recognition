import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import HandGestureDataset
from model import GestureGRU
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():
    # 固定随机种子以确保一致性
    torch.manual_seed(42)
    np.random.seed(42)
    # 加载数据（使用增强数据）
    dataset = HandGestureDataset('../data/hand_gesture_data_augmented.npz')
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 加载模型
    model = GestureGRU()
    model.load_state_dict(torch.load('../models/gesture_gru_cv.pth'))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    # 画混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Hello', 'Thank you', 'Sorry', 'You', 'Goodbye', 'I', 'Love', 'Help', 'Eat', 'Drink'], yticklabels=['Hello', 'Thank you', 'Sorry', 'You', 'Goodbye', 'I', 'Love', 'Help', 'Eat', 'Drink'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("混淆矩阵已保存为 confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()