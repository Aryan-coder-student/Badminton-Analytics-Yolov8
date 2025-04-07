from ultralytics import YOLO
import torch
torch.cuda.empty_cache()
def train_model():
    # Load the model 
    model = YOLO('yolov8n.pt')

    # Train the model on CUDA
    model.train(
        data='data/badminton.v1i.yolov8/data.yaml',
        imgsz=640,
        epochs=10,
        workers=0,
        batch=8,
        name='yolov8n_custom',
    )

if __name__ == '__main__':
    train_model()
