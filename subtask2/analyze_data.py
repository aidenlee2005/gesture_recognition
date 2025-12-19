import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def analyze_data_quality():
    """分析数据质量和分布"""
    print("=== 数据质量分析 ===\n")

    # 加载数据
    try:
        data = np.load('hand_gesture_data.npz')
        X = data['X']
        y = data['y']
        print(f"数据加载成功: {X.shape[0]} 个样本")
    except FileNotFoundError:
        print("❌ 未找到数据文件 'hand_gesture_data.npz'")
        print("请先运行数据收集脚本")
        return

    # 加载类别
    try:
        with open('classes.json', 'r') as f:
            GESTURES = json.load(f)
    except FileNotFoundError:
        print("❌ 未找到类别文件 'classes.json'")
        return

    # 基本统计
    print("📊 基本统计:")
    print(f"  总样本数: {len(X)}")
    print(f"  序列长度: {X.shape[1]} 帧")
    print(f"  特征维度: {X.shape[2]}")
    print(f"  类别数: {len(GESTURES)}")

    # 类别分布
    unique, counts = np.unique(y, return_counts=True)
    print("\n📈 类别分布:")
    for gesture_id, count in zip(unique, counts):
        gesture_name = GESTURES[str(gesture_id)]
        percentage = count / len(X) * 100
        status = "✅" if count >= 10 else "⚠️"
        print(f"  {gesture_id} ({gesture_name}): {count} {status} ({percentage:.1f}%)")
    # 数据质量检查
    print("\n🔍 数据质量检查:")
    # 检查数据完整性
    missing_data = np.isnan(X).any() or np.isinf(X).any()
    print(f"  数据完整性: {'❌ 存在缺失值' if missing_data else '✅ 数据完整'}")

    # 检查数据范围
    data_range = np.ptp(X, axis=(0, 1))  # peak-to-peak along samples and time
    print(f"  特征范围: [{data_range.min():.3f}, {data_range.max():.3f}]")

    # 检查类别平衡
    min_samples = min(counts)
    max_samples = max(counts)
    balance_ratio = min_samples / max_samples
    balance_status = "✅ 平衡" if balance_ratio > 0.7 else "⚠️ 不平衡"
    print(f"  类别平衡: {balance_status} (最小/最大 = {balance_ratio:.2f})")

    # 建议
    print("\n💡 建议:")
    if len(X) < 100:
        print("  ⚠️ 样本数太少，建议至少收集100个样本")
    if min_samples < 10:
        print("  ⚠️ 某些类别样本数太少，建议每个类别至少10个样本")
    if balance_ratio < 0.5:
        print("  ⚠️ 类别分布不平衡，建议平衡各类别样本数")
    if missing_data:
        print("  ❌ 数据存在问题，需要重新收集")

    # 可视化类别分布
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.bar(unique, counts)
    plt.xlabel('Gesture ID')
    plt.ylabel('Sample Count')
    plt.title('Class Distribution')
    plt.xticks(unique)

    # PCA可视化 (取最后一帧)
    plt.subplot(1, 3, 2)
    last_frames = X[:, -1, :]  # 取每个序列的最后一帧
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(last_frames)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Visualization (Last Frame)')
    plt.colorbar(scatter, ticks=unique)

    # 特征方差分析
    plt.subplot(1, 3, 3)
    feature_variance = np.var(X.reshape(-1, X.shape[-1]), axis=0)
    plt.plot(feature_variance)
    plt.xlabel('Feature Index')
    plt.ylabel('Variance')
    plt.title('Feature Variance')
    plt.tight_layout()

    plt.savefig('data_analysis.png', dpi=150, bbox_inches='tight')
    print("\n📊 可视化结果已保存到 'data_analysis.png'")
    plt.show()

def check_model_performance():
    """检查模型性能"""
    print("\n=== 模型性能检查 ===\n")

    try:
        import torch
        from model import GestureGRU
        from dataset import HandGestureDataset
        from torch.utils.data import DataLoader, random_split

        # 加载模型（优先使用交叉验证模型）
        model = GestureGRU()
        try:
            model.load_state_dict(torch.load('gesture_gru_cv.pth'))
            print("加载模型: gesture_gru_cv.pth (交叉验证训练)")
        except:
            try:
                model.load_state_dict(torch.load('gesture_gru.pth'))
                print("加载模型: gesture_gru.pth")
            except:
                print("❌ 未找到训练好的模型")
                return

        model.eval()

        # 加载增强数据进行测试（因为模型是在增强数据上训练的）
        try:
            dataset = HandGestureDataset('hand_gesture_data_augmented.npz')
            print(f"使用增强数据测试 (样本数: {len(dataset)})")
        except:
            # 如果没有增强数据，用原始数据
            dataset = HandGestureDataset('hand_gesture_data.npz')
            print(f"使用原始数据测试 (样本数: {len(dataset)})")
        if len(dataset) < 20:
            print("⚠️ 样本数太少，无法准确评估模型性能")
            return

        # 简单交叉验证
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # 测试
        correct = 0
        total = 0
        predictions = []

        with torch.no_grad():
            for X, y in test_loader:
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                predictions.extend(predicted.tolist())

        accuracy = 100 * correct / total
        print(f"测试准确率: {accuracy:.2f}%")
        if accuracy > 90:
            print("  ✅ 模型在测试集上表现良好")
        elif accuracy > 70:
            print("  ⚠️ 模型表现一般，可能存在过拟合")
        else:
            print("  ❌ 模型表现较差，需要更多数据或调整模型")

    except Exception as e:
        print(f"❌ 模型检查失败: {e}")

if __name__ == "__main__":
    analyze_data_quality()
    check_model_performance()