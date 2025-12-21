import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import HandGestureDataset
from model import GestureGRU
import numpy as np

def train_model():
    # 加载增强后的数据（现在包含标准化）
    dataset = HandGestureDataset('hand_gesture_data_augmented.npz')
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 模型
    model = GestureGRU()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # 添加L2正则化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # 早停
    best_val_acc = 0
    patience = 10
    patience_counter = 0

    # 训练
    num_epochs = 100  # 增加epoch数
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        val_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

        # 学习率调度
        scheduler.step(val_acc)

        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'gesture_gru_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load('gesture_gru_best.pth'))
    # 保存最终模型
    torch.save(model.state_dict(), 'gesture_gru.pth')
    print("最佳模型保存完成")

if __name__ == "__main__":
    train_model()