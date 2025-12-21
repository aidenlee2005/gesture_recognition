import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_3d_hand_keypoints():
    """显示手势手部关键点3D对比图"""
    try:
        img = mpimg.imread('gesture_hand_keypoints_3d.png')
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Gesture Hand Keypoints 3D Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print("❌ 未找到文件: gesture_hand_keypoints_3d.png")
    except Exception as e:
        print(f"❌ 显示图像时出错: {e}")

if __name__ == "__main__":
    print("🎨 显示手势手部关键点3D对比可视化...")
    show_3d_hand_keypoints()