import cv2
import numpy as np
from skimage import feature, exposure
import matplotlib.pyplot as plt
import matplotlib
import os
import skimage

def extract_hog_features(image):
    """
    提取图像的 HOG 特征和 HOG 图像。

    参数:
        image (numpy.ndarray): 输入的图像，可以是灰度图像或彩色图像。

    返回:
        hog_features (numpy.ndarray): HOG 特征向量。
        hog_image_rescaled (numpy.ndarray): 对比度增强后的 HOG 图像。
    """
    # 确定是否为彩色图像
    if image.ndim == 3:
        channel_axis = -1  # 彩色图像的通道轴通常是最后一个轴
    else:
        channel_axis = None  # 灰度图像没有通道轴

    # 计算 HOG 特征和 HOG 图像
    hog_features, hog_image = feature.hog(
        image,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True,
        channel_axis=channel_axis
    )

    # 增强 HOG 图像的对比度以便更好地可视化
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_features, hog_image_rescaled

def display_original_and_hog(image_path):
    """
    读取图像，提取 HOG 特征，并显示原始图像和 HOG 图像。

    参数:
        image_path (str): 图像文件的路径。
    """
    try:
        print(f"正在检查图像路径: {image_path}")
        # 确认图像路径是否存在
        if not os.path.exists(image_path):
            print(f"图像路径不存在: {image_path}")
            return

        # 尝试读取图像（保持原始通道数）
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        # 打印图像信息
        if image is None:
            print(f"无法读取图像文件: {image_path}")
            return
        else:
            print(f"图像读取成功: {image_path}")
            print(f"图像形状: {image.shape}")
            print(f"图像数据类型: {image.dtype}")
            if image.ndim == 2:
                print("图像类型: 灰度图像")
            elif image.ndim == 3:
                print(f"图像类型: 彩色图像，通道数: {image.shape[2]}")
            print(f"图像像素范围: {image.min()} 到 {image.max()}")

        # 确保图像不是全黑
        if np.all(image == 0):
            print("图像数据全为 0，可能读取失败或图像本身全黑。")
            return

        # 如果是彩色图像，转换为 RGB 以便 matplotlib 正确显示
        if image.ndim == 3 and image.shape[2] == 3:
            image_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_display = image

        # 提取 HOG 特征和 HOG 图像
        print("正在提取 HOG 特征...")
        hog_features, hog_image = extract_hog_features(image)
        print("HOG 特征提取完成。")
        print(f"HOG 特征向量长度: {len(hog_features)}")

        # 显示原始图像和 HOG 图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                       sharex=True, sharey=True)

        ax1.axis('off')
        if image_display.ndim == 3:
            ax1.imshow(image_display)
        else:
            ax1.imshow(image_display, cmap='gray')
        ax1.set_title('原始图像')

        ax2.axis('off')
        ax2.imshow(hog_image, cmap='gray')
        ax2.set_title('HOG 增强后的图像')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"发生错误: {e}")

# 设置支持中文的字体
# 根据您的系统和已安装的字体进行调整
matplotlib.rcParams['font.family'] = 'SimHei'  # 适用于 Windows
# matplotlib.rcParams['font.family'] = 'Microsoft YaHei'  # 适用于 Windows
# matplotlib.rcParams['font.family'] = 'PingFang SC'  # 适用于 macOS
# matplotlib.rcParams['font.family'] = 'WenQuanYi'  # 适用于 Linux

# 打印 scikit-image 版本
print(f"scikit-image 版本: {skimage.__version__}")

# 示例用法
if __name__ == "__main__":
    # 使用绝对路径，确保路径正确
    image_path = r".\data\9-1.jpg"  # 替换为您的图像绝对路径
    display_original_and_hog(image_path)
