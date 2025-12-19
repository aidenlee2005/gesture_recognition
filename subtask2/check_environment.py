# 环境兼容性检查脚本

import sys
import platform

def check_environment_compatibility():
    """检查环境兼容性并给出建议"""

    print("🔍 环境兼容性检查")
    print("=" * 50)

    issues = []
    warnings = []

    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        issues.append("Python版本过低，建议使用Python 3.8+")

    # 检查操作系统
    os_name = platform.system()
    print(f"操作系统: {os_name}")
    if os_name not in ["Darwin", "Windows", "Linux"]:
        warnings.append("未在标准操作系统上测试过")

    # 检查关键依赖
    try:
        import cv2
        cv_version = cv2.__version__
        print(f"OpenCV版本: {cv_version}")
        if cv_version < "4.5":
            warnings.append("OpenCV版本较旧，可能影响性能")
    except ImportError:
        issues.append("OpenCV未安装")

    try:
        import mediapipe as mp
        mp_version = mp.__version__
        print(f"MediaPipe版本: {mp_version}")

        # 检查API类型
        if hasattr(mp, 'tasks'):
            print("MediaPipe API: 新版tasks API")
            warnings.append("使用新版MediaPipe API，可能需要更新代码")
        elif hasattr(mp, 'solutions'):
            print("MediaPipe API: 旧版solutions API")
        else:
            issues.append("MediaPipe API类型未知")

        if mp_version != "0.10.31":
            warnings.append(f"MediaPipe版本与开发环境不同 (开发: 0.10.31, 当前: {mp_version})")

    except ImportError:
        issues.append("MediaPipe未安装")

    try:
        import torch
        torch_version = torch.__version__
        print(f"PyTorch版本: {torch_version}")
        if torch_version < "1.9":
            warnings.append("PyTorch版本较旧")
    except ImportError:
        issues.append("PyTorch未安装")

    try:
        import numpy as np
        np_version = np.__version__
        print(f"NumPy版本: {np_version}")
    except ImportError:
        issues.append("NumPy未安装")

    try:
        import sklearn
        sklearn_version = sklearn.__version__
        print(f"Scikit-learn版本: {sklearn_version}")
    except ImportError:
        issues.append("Scikit-learn未安装")

    print("\n" + "=" * 50)

    if issues:
        print("❌ 严重问题:")
        for issue in issues:
            print(f"  • {issue}")

    if warnings:
        print("⚠️  警告:")
        for warning in warnings:
            print(f"  • {warning}")

    if not issues and not warnings:
        print("✅ 环境兼容性良好")
    elif not issues:
        print("✅ 无严重问题，但请注意上述警告")
    else:
        print("❌ 存在严重问题，需要解决后才能正常运行")

    return len(issues) == 0

if __name__ == "__main__":
    check_environment_compatibility()