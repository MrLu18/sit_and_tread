import os
import cv2
import time
import numpy as np
from PIL import Image, ImageDraw,ImageFont
from ultralytics import YOLO
from glob import glob


# 计算IoU
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
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box1[1])

    # 计算并集面积
    area_union = area_box1 + area_box2 - area_int

    # 计算IoU
    iou = area_int / area_union if area_union > 0 else 0
    return iou


def check_iou_between_person_and_targets(video_path):
    model = YOLO('/mnt/jrwbxx/yolo11/runs/detect/train5/weights/12_24_sit_and_tread.pt')
    target_classes = [1, 2, 3]  # bag, box, cart
    cls4 = 4  # 新增加的类别编号（假设为 4）

    # 获取当前时间
    current_time = time.localtime()
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)

    # 输出目录
    #output_dir = '/mnt/jrwbxx/yolo11/output_sit_and_tread'
    output_dir1 = '/mnt/jrwbxx/yolo11/experience1'
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1, exist_ok=True)

    # 打开视频流
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    frame_interval = int(fps)  # 每秒提取一帧

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break  # 视频结束

        # 每秒提取一帧
        if frame_idx % frame_interval == 0:
            # 使用YOLO进行物体检测
            results = model(frame, device=1)  # 直接输入图像进行检测
            boxes = results[0].boxes

            person_boxes = []
            target_boxes = []
            cls4_boxes = []

            # 处理检测结果
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                con = boxes.conf[i].item()

                if cls == 0 and con > 0.1:  # 如果是 "person" 类别
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    person_boxes.append([x1, y1, x2, y2])

                if cls in target_classes and con > 0.1:  # 目标类别（bag, box, cart）
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    target_boxes.append([x1, y1, x2, y2])

                if cls == cls4 and con > 0.1:  # 如果是 "cls 4" 类别
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    cls4_boxes.append([x1, y1, x2, y2])

            # 如果检测到 "person" 类别和目标框，计算IoU
            if person_boxes and target_boxes:
                for i, person_box in enumerate(person_boxes):
                    person_center_y = (person_box[1] + person_box[3]) / 2

                    for j, target_box in enumerate(target_boxes):
                        target_center_y = (target_box[1] + target_box[3]) / 2

                        # 判断目标框的中心点是否在person框的下方
                        if target_center_y + 20 > person_center_y:  # 根据需要调整此阈值
                            iou = calculate_iou(person_box, target_box)

                            if iou > 0:  # IoU大于0，进入新的判断
                                # 创建一个包围person和target框的区域 本来思路是判断脚是否在这个大框框里面 但是这个不合理 应该判断脚是不是在这个交集里面，再检测脚和人与物交集有没有交集
                                min_x = max(person_box[0], target_box[0])
                                min_y = max(person_box[1], target_box[1])
                                max_x = min(person_box[2], target_box[2])
                                max_y = min(person_box[3], target_box[3])

                                target_area = (min_x, min_y, max_x, max_y)


                                # 检查是否有cls 4物体在这个区域内
                                for cls4_box in cls4_boxes: #比较巧妙 提前存储这个变量 然后检测这个物体是不是在这里面
                                    # 判断cls 4框是否与这个区域有交集
                                    inter_x1 = max(cls4_box[0], target_area[0])
                                    inter_y1 = max(cls4_box[1], target_area[1])
                                    inter_x2 = min(cls4_box[2], target_area[2])
                                    inter_y2 = min(cls4_box[3], target_area[3])

                                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                                        # 计算交集区域的面积
                                        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

                                        # 计算 cls4_box 的面积
                                        cls4_area = (cls4_box[2] - cls4_box[0]) * (cls4_box[3] - cls4_box[1])

                                        # 计算交集区域占 cls4_box 面积的比例
                                        overlap_ratio = intersection_area / cls4_area



                                        # 如果占比大于某个阈值，可以执行进一步的操作
                                        if overlap_ratio > 0.5:

                                            # 在此处可以绘制标注或进行其他操作

                                            # 加载图像并绘制标注

                                            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #注意 cap.read 获取到的帧是numpy形式 不能直接配合Draw函数 需要转换

                                            # 创建绘图上下文
                                            draw = ImageDraw.Draw(img)

                                            # 继续进行绘制操作

                                            # 绘制 "person" 类别框
                                            draw.rectangle([person_box[0], person_box[1], person_box[2], person_box[3]],
                                                           outline="red", width=3)


                                            # 绘制目标框（bag, box, cart）
                                            draw.rectangle([target_box[0], target_box[1], target_box[2], target_box[3]],
                                                           outline="blue", width=3)

                                            # 绘制 cls 4 类别框
                                            draw.rectangle([cls4_box[0], cls4_box[1], cls4_box[2], cls4_box[3]],
                                                           outline="green", width=3)
                                            # 在框附近绘制 IOU 的值
                                            font = ImageFont.load_default()  # 使用默认字体
                                            text = f"IOU: {overlap_ratio:.2f}"
                                            text_position = ( inter_x1,  inter_y1)  # 在框的上方显示 IOU 值
                                            draw.text(text_position, text, fill="yellow", font=font)
                                            # 保存标注后的图像
                                            saved_image_path = os.path.join(output_dir1, f"{timestamp}_{frame_idx}_detected.jpg")
                                            img.save(saved_image_path)


        frame_idx += 1
    cap.release()




if __name__ == '__main__':

    video_path = '/mnt/jrwbxx/yolo11/s_and_t_1_12.mp4'  # 输入你的视频路径
    check_iou_between_person_and_targets(video_path)
