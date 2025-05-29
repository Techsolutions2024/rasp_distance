from ultralytics import YOLO


model = YOLO("best_int8_openvino_model/")

# In ra tất cả class mà mô hình hỗ trợ
for class_id, class_name in model.names.items():
    print(f"ID {class_id}: {class_name}")
