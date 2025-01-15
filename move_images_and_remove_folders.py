import os
import shutil


def move_images_and_remove_folders(root_dir):
    """
    将指定目录下所有子文件夹中的图像文件移动到根目录，并删除空文件夹。

    参数：
    root_dir (str): 根目录路径。
    """
    # 获取根目录下的所有子文件夹
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        # 确保是文件夹
        if os.path.isdir(folder_path):
            # 遍历该文件夹中的所有文件
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                # 如果是图像文件（根据扩展名判断）
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    # 将图像文件移动到根目录
                    shutil.move(file_path, os.path.join(root_dir, file_name))

            # 删除空文件夹
            if not os.listdir(folder_path):  # 如果文件夹为空
                os.rmdir(folder_path)
                print(f"已删除空文件夹: {folder_path}")


# 示例使用
root_directory = r'D:\python_program\yolo11\2024'  # 请替换为实际路径
move_images_and_remove_folders(root_directory)
