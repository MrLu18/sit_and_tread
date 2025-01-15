import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from pathlib import Path
import os

# 加载 YOLO 模型
model = YOLO('/mnt/jrwbxx/yolo11/runs/detect/train3/weights/12_18_bag.pt')

def process_video_stream(video_source=0, output_folder=None, target_classes=[0, 1]):
    """
    实时处理视频流，每秒检测一帧，检测目标并在帧中标注。

    参数：
    video_source (int/str): 视频源，0 表示默认摄像头，也可为视频文件路径。
    output_folder (str): 可选，保存标注帧的文件夹路径。
    target_classes (list): 要检测的目标类别索引列表（默认为 [0, 1]）。
    """
    # 创建输出文件夹（如果需要保存标注帧）
    if output_folder:
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    # 打开视频流
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("无法打开视频流。")
        return

    # 获取视频的帧率
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps  # 每秒检测一帧

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频流结束或读取错误。")
            break

        # 每隔指定帧检测一次
        if frame_count % frame_interval == 0:
            # 转换为 PIL 图像格式
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 使用 YOLO 模型进行检测
            results = model(img)
            boxes = results[0].boxes

            # 创建一个绘图对象
            draw = ImageDraw.Draw(img)
            detected = False

            # 遍历检测结果
            for box in boxes:
                cls = int(box.cls.item())  # 获取类别索引
                if cls in target_classes:
                    detected = True
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # 获取边界框坐标

                    # 在图像上绘制边界框和类别名称
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    draw.text((x1, y1 - 10), f"Class {cls}", fill="red")

            # 转换回 OpenCV 格式
            annotated_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # 显示标注后的帧
            # cv2.imshow("Video Stream", annotated_frame)

            # 如果需要保存标注帧
            if detected and output_folder:
                frame_name = f"frame_{saved_frame_count:05d}.jpg"
                frame_path = os.path.join(output_folder, frame_name)
                cv2.imwrite(frame_path, annotated_frame)
                saved_frame_count += 1

        frame_count += 1

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"检测完成，共保存 {saved_frame_count} 张标注帧。")

# 示例使用
process_video_stream(video_source='/mnt/jrwbxx/yolo11/experience', output_folder='/mnt/jrwbxx/yolo11/output_sit_and_tread') #video_source 为0则表示使用摄像头 也可以为视频文件
