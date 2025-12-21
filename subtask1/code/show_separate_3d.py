import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_separate_3d_hand_keypoints():
    """显示分开并排的手势手部关键点3D对比图"""
    try:
        img = mpimg.imread('gesture_hand_keypoints_3d_separate.png')
        plt.figure(figsize=(20, 12))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Gesture Hand Keypoints 3D - Separate Views', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print("❌ 未找到文件: gesture_hand_keypoints_3d_separate.png")
    except Exception as e:
        print(f"❌ 显示图像时出错: {e}")

if __name__ == "__main__":
    print("🎨 显示分开并排的手势手部关键点3D对比可视化...")
    show_separate_3d_hand_keypoints()