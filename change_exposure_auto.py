import cv2
import numpy as np
import os


def adjust_exposure(image, target_brightness=10):
    # 计算图像的当前平均亮度
    current_brightness = np.mean(image)
    print("Current brightness:", current_brightness)

    # 如果图像曝光过低，则增加亮度
    if current_brightness < target_brightness:
        # 计算亮度差异并调整图像曝光
        brightness_diff = target_brightness - current_brightness
        image = cv2.convertScaleAbs(image, alpha=1, beta=brightness_diff)  # alpha 对比度 beta 亮度
    else:
        brightness_diff = target_brightness - current_brightness
        image = cv2.convertScaleAbs(image, alpha=1, beta=brightness_diff)

    return image


def adjust_images_in_folder(folder_path, target_brightness=10):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 获取完整的文件路径
        file_path = os.path.join(folder_path, filename)

        # 仅处理图像文件
        if os.path.isfile(file_path) and (filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))):
            # 加载图像
            image = cv2.imread(file_path)

            if image is None:
                print(f"无法加载图像: {file_path}")
                continue

            # 调整曝光度
            adjusted_image = adjust_exposure(image, target_brightness)

            # 保存调整后的图像
            output_path = os.path.join(folder_path, f"adjusted_{filename}")
            cv2.imwrite(output_path, adjusted_image)
            print(f"已保存调整后的图像: {output_path}")


# 指定文件夹路径
folder_path = r'D:\python_program\yolo11\dataset\12\01'  # 替换为你的文件夹路径

# 调整该文件夹内所有图像的曝光度
adjust_images_in_folder(folder_path, target_brightness=10)
