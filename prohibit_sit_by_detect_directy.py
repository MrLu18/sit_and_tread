import os
import cv2
import time
from ultralytics import YOLO
from PIL import Image, ImageDraw,ImageFont

def check_person_sit(video_path, output_dir="/mnt/jrwbxx/yolo11/saved_frames/"):
    model = YOLO('/mnt/jrwbxx/yolo11/runs/detect/train11/weights/best.pt')

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # 每秒处理的帧间隔
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            results = model(frame, device=0, verbose=False, save=False)  # 关闭自动保存
            boxes = results[0].boxes

            detected_boxes = []
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                conf = boxes.conf[i].item()
                if cls == 0 and conf > 0.1:
                    # 转换为像素坐标
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    detected_boxes.append((x1, y1, x2, y2))

            if detected_boxes:
                # 绘制矩形框并保存图像
                img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img)
                for detected_boxe in detected_boxes:
                    draw.rectangle([detected_boxe[0],detected_boxe[1],detected_boxe[2],detected_boxe[3]],outline="red", width=3)
                # 生成时间戳文件名
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                output_path = os.path.join(output_dir, f"{timestamp}_{frame_idx}.jpg")
                os.makedirs(output_dir, exist_ok=True)
                img.save(output_path)

        frame_idx += 1
    cap.release()

if __name__ == '__main__':
    video_path = '/mnt/jrwbxx/yolo11/experiment.mp4'
    check_person_sit(video_path)