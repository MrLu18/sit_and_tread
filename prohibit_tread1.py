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
def check_cls4_overlap(person_box, target_box, cls4_boxes):
    """
    检查是否有 cls4 物体在目标框与人物框交集区域内，并计算交集面积与 cls4 物体面积的重叠比率。
    返回重叠比率最大值以及其对应的交集区域坐标和 cls4 框。

    参数:
    person_box (tuple): 人物框 (x_min, y_min, x_max, y_max)
    target_box (tuple): 目标框 (x_min, y_min, x_max, y_max)
    cls4_boxes (list): cls4 物体框的列表，每个框为 (x_min, y_min, x_max, y_max)

    返回:
    tuple:
        - 最大的重叠比率 (float)，如果没有符合条件的物体则为 None。
        - 最大重叠比率对应的交集区域的坐标 (inter_x1, inter_y1)。
        - 对应的 cls4 框 (tuple)，如果没有符合条件的物体则为 None。
    """
    # 计算目标框与人物框的交集区域
    min_x = max(person_box[0], target_box[0])
    min_y = max(person_box[1], target_box[1])
    max_x = min(person_box[2], target_box[2])
    max_y = min(person_box[3], target_box[3])

    target_area = (min_x, min_y, max_x, max_y)  # 交集区域

    max_overlap_ratio = 0  # 用于存储最大的重叠比率
    best_inter_x1 = 0  # 用于存储最大重叠比率对应的交集坐标
    best_inter_y1 = 0
    best_cls4_box = 0  # 用于存储对应的 cls4 框

    # 检查是否有 cls4 物体在交集区域内
    for cls4_box in cls4_boxes:
        # 判断 cls4 框是否与交集区域有交集
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

            # 更新最大重叠比率及其相关信息
            if max_overlap_ratio == 0 or overlap_ratio > max_overlap_ratio:
                max_overlap_ratio = overlap_ratio
                best_inter_x1 = inter_x1
                best_inter_y1 = inter_y1
                best_cls4_box = cls4_box

    return max_overlap_ratio, (best_inter_x1, best_inter_y1), best_cls4_box


def check_iou_between_person_and_targets(video_path):
    model = YOLO('/mnt/jrwbxx/yolo11/runs/detect/train5/weights/12_24_sit_and_tread.pt')
    #target_classes = [1, 2]  # bag, box  针对1 2多一层处理 不需要识别它了
    cls3 =3 #对cart单独一类
    cls4 = 4  # 新增加的类别编号（假设为 4）

    # 获取当前时间
    current_time = time.localtime()
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)

    # 输出目录
    #保存我想要的输出 项目不需要
    output_file = "mnt/jrwbxx/yolo11/outfile.txt"
    output_dir = '/mnt/jrwbxx/yolo11/output_sit_and_tread'
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
            # 保存当前帧的路径
            # frame_path = os.path.join(output_dir, f"frame_{timestamp}_{frame_idx}.jpg")
            # cv2.imwrite(frame_path, frame)
            # 使用YOLO进行物体检测
            results = model(frame, device=1)  # 直接输入图像进行检测
            boxes = results[0].boxes

            person_boxes = []
            cls3_boxes = []
            cls4_boxes = []

            # 处理检测结果
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                con = boxes.conf[i].item()

                if cls == 0 and con > 0.1:  # 如果是 "person" 类别  先把对应类别添加上去  先把每一个识别物体输入上去
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    person_boxes.append([x1, y1, x2, y2])

                if cls == cls3 and con > 0.1:  # 目标类别 cart
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    cls3_boxes.append([x1, y1, x2, y2])

                if cls == cls4 and con > 0.1:  # 如果是 "cls 4" 类别
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    cls4_boxes.append([x1, y1, x2, y2])

                # img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #注意 cap.read 获取到的帧是numpy形式 不能直接配合Draw函数
                #  注意这部分逻辑要改 因为我加了一个目标
                #
                #
                # # 创建绘图上下文
                # draw = ImageDraw.Draw(img) #实验一下 能不能检测到
                #
                # # 继续进行绘制操作
                # for i, person_box in enumerate(person_boxes):
                #     for j, target_box in enumerate(target_boxes):
                #         for cls4_box in cls4_boxes:
                #             print(frame1) #有点奇怪  为什么有些没有被识别到
                # # 绘制 "person" 类别框
                #             draw.rectangle([person_box[0], person_box[1], person_box[2], person_box[3]],
                #                            outline="red", width=3)
                #
                #             # 绘制目标框（bag, box, cart）
                #             draw.rectangle([target_box[0], target_box[1], target_box[2], target_box[3]],
                #                            outline="blue", width=3)
                #
                #             # 绘制 cls 4 类别框
                #             draw.rectangle([cls4_box[0], cls4_box[1], cls4_box[2], cls4_box[3]],
                #                            outline="green", width=3)
                #             saved_image_path = os.path.join(output_dir, f"{frame1}_detected.jpg")
                #             img.save(saved_image_path)
                #             print("输出保存的路径",saved_image_path)
                # frame1 = frame1+1

            # 如果检测到 "person" 类别和目标框，计算IoU
            if person_boxes and cls3_boxes: # 对每一个都进行计算 哪一个成功就输出哪一个 先统计车子 看看有没有报警 车子不考虑高度
                for i, person_box in enumerate(person_boxes):

                    for j, cls3_box in enumerate(cls3_boxes):

                        # 判断目标框的中心点是否在person框的下方
                            iou = calculate_iou(person_box, cls3_box)
                         # 踩的逻辑 一个是要求车子和人有交并比 其次判断脚是不是在这个交并比内 并且比重不能太低
                            if iou > 0.2: # 用来判断坐 如果iou足够高 就视为是坐
                                # 加载图像并绘制标注
                                # 看看这个值大概有多少 好衡量一下
                                with open(output_file, "w", encoding="utf-8") as file:
                                    file.write(f"这个iou大于0.2的 这是第{frame_idx}\n")  # 写入介绍
                                    file.write(iou)

                                img = Image.fromarray(cv2.cvtColor(frame,
                                                                   cv2.COLOR_BGR2RGB))  # 注意 cap.read 获取到的帧是numpy形式 不能直接配合Draw函数 需要转换

                                # 创建绘图上下文
                                draw = ImageDraw.Draw(img)

                                # 继续进行绘制操作

                                # 绘制 "person" 类别框
                                draw.rectangle([person_box[0], person_box[1], person_box[2], person_box[3]],
                                               outline="red", width=3)

                                # 绘制目标框（bag, box, cart）
                                draw.rectangle([cls3_box[0], cls3_box[1], cls3_box[2], cls3_box[3]],
                                               outline="blue", width=3)

                                # 绘制 cls 4 类别框
                                # draw.rectangle([cls4_box[0], cls4_box[1], cls4_box[2], cls4_box[3]],
                                #                outline="green", width=3)
                                # 保存标注后的图像
                                saved_image_path = os.path.join(output_dir1,
                                                                f"{timestamp}_{frame_idx}_detected.jpg")
                                img.save(saved_image_path)
                                print(f"{saved_image_path}")
                            elif iou > 0 :  # IoU大于0，进入新的判断
                                # 创建一个包围person和target框的区域 本来思路是判断脚是否在这个大框框里面 但是这个不合理 应该判断脚是不是在这个交集里面，再检测脚和人与物交集有没有交集

                                overlap_ratio, inter_xy, best_cls4_box = check_cls4_overlap(person_box, cls3_box,cls4_boxes)

                                inter_x1, inter_y1 = inter_xy

                                # 如果占比大于某个阈值，可以执行进一步的操作
                                if overlap_ratio > 0.2:  # 暂定0.2 可以增加

                                    # 加载图像并绘制标注

                                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 注意 cap.read 获取到的帧是numpy形式 不能直接配合Draw函数 需要转换

                                    # 创建绘图上下文
                                    draw = ImageDraw.Draw(img)

                                    # 继续进行绘制操作

                                    # 绘制 "person" 类别框
                                    draw.rectangle([person_box[0], person_box[1], person_box[2], person_box[3]],
                                                   outline="red", width=3)

                                    # 绘制目标框（bag, box, cart）
                                    draw.rectangle([cls3_box[0], cls3_box[1], cls3_box[2], cls3_box[3]],
                                                   outline="blue", width=3)

                                    # 绘制 cls 4 类别框
                                    # draw.rectangle([cls4_box[0], cls4_box[1], cls4_box[2], cls4_box[3]],
                                    #                outline="green", width=3)
                                    # 在框附近绘制 IOU 的值
                                    font = ImageFont.load_default()  # 使用默认字体
                                    text = f"IOU: {overlap_ratio:.2f}"
                                    text_position = (inter_x1, inter_y1)  # 在框的上方显示 IOU 值
                                    draw.text(text_position, text, fill="yellow", font=font)
                                    # 保存标注后的图像
                                    saved_image_path = os.path.join(output_dir1,
                                                                    f"{timestamp}_{frame_idx}_detected.jpg")
                                    img.save(saved_image_path)
                                    print(f"{saved_image_path}")



        frame_idx += 1
    cap.release()




if __name__ == '__main__':

    video_path = '/mnt/jrwbxx/yolo11/1.mp4'  # 输入你的视频路径
    check_iou_between_person_and_targets(video_path)
