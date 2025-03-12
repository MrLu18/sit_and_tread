import os
import cv2
import time
from ultralytics import YOLO
from PIL import Image, ImageDraw

def calculate_iou(box1, box2):
    # 计算交集区域 也就是找到次左上 次右下
    x1_int = max(box1[0], box2[0])
    y1_int = max(box1[1], box2[1])
    x2_int = min(box1[2], box2[2])
    y2_int = min(box1[3], box2[3])

    # 交集区域的宽高
    width_int = max(0, x2_int - x1_int)
    height_int = max(0, y2_int - y1_int)
    area_int = width_int * height_int

    # 计算两个框的面积
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    area_union = area_box1 + area_box2 - area_int

    # 计算IoU
    iou = area_int / area_union if area_union > 0 else 0
    return iou

def check_person_sit(video_path, output_dir="/mnt/jrwbxx/yolo11/saved_frames/"):
    model = YOLO('/mnt/jrwbxx/yolo11/runs/detect/train/weights/best.pt')  # 区域检测模型
    model1 = YOLO('yolo11m.pt')  # 人体检测模型

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # 每秒处理1帧
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            #获取当前图像的尺寸
            height, width = frame.shape[:2]
            right_threshold = width*0.9

            # 执行推理
            results = model(frame, device=0, verbose=False, save=False)
            results1 = model1(frame, device=0, verbose=False, save=False)

            # 提取区域框（假设类别0是目标区域）
            area_boxes = [
                list(map(int, box.xyxy[0].tolist()))
                for box in results[0].boxes
                if box.cls == 0 and box.conf > 0.1
            ]

            # 提取人体框（假设类别0是人）
            human_boxes = [
                list(map(int, box.xyxy[0].tolist()))
                for box in results1[0].boxes
                if box.cls == 0 and box.conf > 0.1
            ]

            # 记录需要绘制的有效框对
            valid_pairs = []
            for a_box in area_boxes:
                a_x1, a_y1, a_x2, a_y2 = a_box
                for h_box in human_boxes:
                    h_x1, h_y1, h_x2, h_y2 = h_box
                    # 检查包含关系
                    iou = calculate_iou(a_box,h_box )  #要求有一定的iou 并且面积不能太小 避免非人误报和只检测到人体单个部位

                    if iou >0.7:
                        #计算区域中心点
                        a_center = (a_x1 + a_x2)/2

                        #排除右侧百分之十的区域
                        if a_center >= right_threshold:
                            continue

                        a_box_area = (a_x1-a_x2)*(a_y1-a_y2)
                        h_box_area = (h_x1-h_x2)*(h_y1-h_y2)
                        if a_box_area > h_box_area * 0.8:
                            valid_pairs.append(a_box) #可以添加单个元素比如元组 字典等 一次性
                            break #有一个则足够了 不需要那么多



            # 仅当存在有效框对时才保存图像
            if valid_pairs:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img)

                # 绘制所有有效框对
                for a_box in valid_pairs:
                    # 绘制区域框（红色）
                    draw.rectangle(a_box, outline="red", width=3)
                    # 绘制人体框（绿色）

                # 生成文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    output_dir,
                    f"{timestamp}_{frame_idx}.jpg"
                )
                os.makedirs(output_dir, exist_ok=True)
                img.save(output_path)
                print(f"Saved: {output_path}")

        frame_idx += 1

    cap.release()


if __name__ == '__main__':
    video_path = '/mnt/jrwbxx/yolo11/save_frames/2.mp4'
    check_person_sit(video_path)