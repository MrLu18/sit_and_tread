import os
from ultralytics import YOLO
import cv2


def detect_and_save(src_directory, dest_directory, cropped_directory):
    # 创建目标文件夹，如果不存在的话
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    if not os.path.exists(cropped_directory):
        os.makedirs(cropped_directory)

    # 加载YOLOv8模型，使用预训练的yolov8模型
    model = YOLO("yolov8n.pt")  # 你可以根据需要使用其他模型，例如"yolov8s.pt"

    # 遍历源文件夹中的所有图像文件
    for root, dirs, files in os.walk(src_directory):
        for file in files:
            # 只处理图像文件（可以根据需要处理其他类型的文件）
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(root, file)

                # 读取图像
                img = cv2.imread(image_path)

                # 使用YOLOv8进行推理（检测目标）
                results = model(img)

                # 获取检测结果（检测到的人类类标签是0） 应该是yolo官方自带的
                boxes = results[0].boxes.xyxy
                confidences = results[0].boxes.conf
                classes = results[0].boxes.cls
                for i in range(len(boxes)):
                    cls = int(classes[i].item())
                    label = model.names[cls]
                    if label == "person":  # 检测到的是“人”类
                        # 获取目标框的坐标 [x1, y1, x2, y2]
                        x1, y1, x2, y2 = boxes[i].tolist()
                        # 在原图上保存检测到的目标框
                        result_image_path = os.path.join(dest_directory, file)
                        cv2.imwrite(result_image_path, img)
                        print(f"Saved image with person: {result_image_path}")

                        # 裁剪出检测到的目标区域
                        cropped_img = img[int(y1):int(y2), int(x1):int(x2)]

                        # 保存裁剪后的目标区域
                        cropped_image_path = os.path.join(cropped_directory, f"cropped_{file}_{i}.jpg")
                        cv2.imwrite(cropped_image_path, cropped_img)
                        print(f"Saved cropped person image: {cropped_image_path}")
                    else:
                        print(f"No person detected in: {image_path}")


# 使用示例
src_directory = '/mnt/jrwbxx/yolo11/experience'  # 替换为包含图像的文件夹路径
dest_directory = '/mnt/jrwbxx/yolo11/experience1'  # 替换为保存图像的目标文件夹路径
cropped_directory = '/mnt/jrwbxx/yolo11/experience2'  # 替换为保存裁剪图像的目标文件夹路径

detect_and_save(src_directory, dest_directory, cropped_directory)
