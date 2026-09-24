from ultralytics import YOLO

model_path = "/opt/infer_cam_calibrator/models/cam_calibrator_1_1_9.pt"
model = YOLO(model_path)  

# Export the model
model.export(
    format="onnx",
    imgsz=2592,
    device="cpu",
    task="detect",
)