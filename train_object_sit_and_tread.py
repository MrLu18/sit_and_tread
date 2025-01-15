from ultralytics import YOLO
import torch

torch.cuda.empty_cache()
# Load a model
def train_yolo11():
    # model = YOLO("yolo11m.yaml").load("yolo11m.pt")  # build a new model from scratch -p2是用于小目标检测
    # #model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # #model = YOLO(r"/mnt/jrwbxx/yolov8/runs/detect/build_pro1/weights/best.pt")  # load a pretrained model (recommended for training) 如果想从头开始训练一个权重文件 就不需要这个，如果是微调，那么可以加入这个
    #
    # # Use the model
    # model.train(data="/mnt/jrwbxx/yolo11/persontrenew.yaml", epochs=100,batch=6,imgsz=1280,device=1)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set

    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format
    #用于被打断的训练 继续之前的训练

    model = YOLO("/mnt/jrwbxx/yolo11/runs/detect/train5/weights/last.pt")
    results = model.train(resume=True)
if __name__ == '__main__':
    train_yolo11()