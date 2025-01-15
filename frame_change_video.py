import cv2
import os

def images_to_video(image_folder, output_video_path, frame_rate=30):
    """
    将指定文件夹中的多张图像转换为视频流，按每秒给定的帧率保存。

    参数：
    image_folder (str): 包含图像的文件夹路径。
    output_video_path (str): 输出视频的保存路径。
    frame_rate (int): 每秒保存的帧数，默认为10帧。
    """
    # 获取文件夹中的所有图像文件
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    # 如果图像文件夹为空
    if not image_files:
        print("没有找到图像文件。")
        return

    # 按文件名排序
    image_files.sort()

    # 读取第一张图像来确定视频的分辨率
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # 设置视频编写器，指定视频输出路径和分辨率
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用XVID编码
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # 将图像按顺序写入视频
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path)

        # 将图像写入视频文件
        video_writer.write(img)

    # 释放视频编写器
    video_writer.release()
    print(f"视频已保存到 {output_video_path}")

# 使用示例
image_folder = '/mnt/jrwbxx/yolo11/dataset/2024'  # 替换为包含图像的文件夹路径
output_video_path = '/mnt/jrwbxx/yolo11/S_and_T.mp4'  # 替换为输出视频的路径

# 调用函数转换图像为视频
images_to_video(image_folder, output_video_path, frame_rate=1) #自己修改帧率
