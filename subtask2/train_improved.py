import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import HandGestureDataset
from model import GestureGRU
import numpy as np
from sklearn.model_selection import KFold

def train_with_cross_validation():
    """使用交叉验证训练，避免过拟合"""
    print("=== 交叉验证训练 ===\n")

    # 加载数据
    try:
        dataset = HandGestureDataset('hand_gesture_data.npz')
        print(f"加载数据: {len(dataset)} 个样本")
    except:
        print("❌ 请先收集数据")
        return

    if len(dataset) < 50:
        print("⚠️ 样本数太少，建议至少50个样本")
        return

    # K折交叉验证
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_accuracies = []
    best_model_state = None
    best_accuracy = 0

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f"\n--- Fold {fold+1}/{k} ---")

        # 创建数据子集
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

        # 初始化模型
        model = GestureGRU()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # 训练
        num_epochs = 50
        best_val_acc = 0
        patience = 10
        patience_counter = 0

        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            for X, y in train_loader:
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 验证阶段
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

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
            # 早停
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        fold_accuracies.append(best_val_acc)
        print(f"Fold {fold+1} best accuracy: {best_val_acc:.2f}%")
        if best_val_acc > best_accuracy:
            best_accuracy = best_val_acc
            best_model_state = model.state_dict()

    # 保存最佳模型
    if best_model_state:
        torch.save(best_model_state, 'gesture_gru_cv.pth')
        print("\n✅ 最佳模型已保存为 'gesture_gru_cv.pth'")    # 统计结果
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print("\n📊 交叉验证结果:")
    print(f"  平均准确率: {mean_acc:.2f}%")
    print(f"  标准差: {std_acc:.2f}%")
    if mean_acc > 80:
        print("✅ 模型泛化能力良好")
    elif mean_acc > 60:
        print("⚠️ 模型泛化能力一般，可能需要更多数据")
    else:
        print("❌ 模型泛化能力较差，强烈建议收集更多数据")

def train_with_data_augmentation():
    """使用轻度数据增强训练"""
    print("\n=== 轻度数据增强训练 ===\n")

    # 加载原始数据
    try:
        dataset = HandGestureDataset('hand_gesture_data.npz')
        print(f"加载数据: {len(dataset)} 个样本")
    except:
        print("❌ 请先收集数据")
        return

    # 轻度增强：每个样本生成2-3个变体
    from augment_data import augment_dataset
    X_aug, y_aug = augment_dataset(dataset.X, dataset.y, augment_factor=2)
    print(f"轻度增强后: {len(X_aug)} 个样本")

    # 保存轻度增强数据
    np.savez('hand_gesture_data_light_aug.npz', X=X_aug, y=y_aug)

    # 训练
    aug_dataset = HandGestureDataset('hand_gesture_data_light_aug.npz')
    train_size = int(0.8 * len(aug_dataset))
    val_size = len(aug_dataset) - train_size
    train_dataset, val_dataset = random_split(aug_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = GestureGRU()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    num_epochs = 100
    best_val_acc = 0

    for epoch in range(num_epochs):
        # 训练
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
        scheduler.step(val_acc)

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'gesture_gru_light_aug.pth')

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("✅ 轻度增强模型已保存为 'gesture_gru_light_aug.pth'")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'cv':
        train_with_cross_validation()
    elif len(sys.argv) > 1 and sys.argv[1] == 'aug':
        train_with_data_augmentation()
    else:
        print("使用方法:")
        print("  python train_improved.py cv    # 交叉验证训练")
        print("  python train_improved.py aug   # 轻度增强训练")