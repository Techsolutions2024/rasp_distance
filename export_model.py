from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO("best.pt")
# Export the model
model.export(format="openvino", half = True, int8 = True)  # creates 'yolov8n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO("best_int8_openvino_model/")
# Export the model to NCNN format
#model.export(format="ncnn", half = True)  # creates '/yolo11n_ncnn_model'

# Load the exported NCNN model
#ncnn_model = YOLO("./yolov8n_ncnn_model")

# Run inference
#results = ncnn_model("https://ultralytics.com/images/bus.jpg")
# Run inference
#results = model("https://ultralytics.com/images/bus.jpg")

# Run inference with specified device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
#results = model("https://ultralytics.com/images/bus.jpg", device="intel:gpu")