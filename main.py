import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO

# --- Thông số hiệu chỉnh khoảng cách ---
ALPHA = 0.8  # Hệ số hiệu chỉnh khoảng cách

# Load thông số hiệu chỉnh camera
data = np.load("camera_calib.npz")
mtx = data["camera_matrix"]
dist = data["dist_coeffs"]

# Tính tiêu cự trung bình từ ma trận camera
f_x = mtx[0, 0]
f_y = mtx[1, 1]
focal_length = (f_x + f_y) / 2  # đơn vị pixel

REAL_HEIGHTS = {
    0: 1.6,   # laixe
    1: 1.6,   # nguoi
    2: 4.0,   # tauhoa
    3: 3.2,   # xebuyt
    4: 1.1,   # xedap
    5: 4.2,   # xedaukeo
    6: 3.2,   # xedulich
    7: 1.5,   # xemay
    8: 1.5,   # xeoto
    9: 3.0    # xetai
}


# Load mô hình YOLOv8
model = YOLO("best_int8_openvino_model/")

# Biến global để chia sẻ frame
latest_frame = None
lock = threading.Lock()

# Skip config: xử lý mỗi N frame
SKIP_FRAME = 2
frame_index = 0

# Thread đọc webcam
def camera_thread(cap):
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with lock:
            latest_frame = frame.copy()

# Mở webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Không thể mở webcam")
    exit()

# Khởi động thread camera
t = threading.Thread(target=camera_thread, args=(cap,))
t.daemon = True
t.start()

# Biến FPS tổng thể
frame_count = 0
start_time = time.time()
fps = 0

# Biến FPS YOLO
yolo_fps = 0
yolo_infer_times = []

annotated_frame = None  # Lưu frame đã gắn nhãn

while True:
    with lock:
        frame = latest_frame.copy() if latest_frame is not None else None

    if frame is None:
        continue

    frame_index += 1

    # Hiệu chỉnh ảnh (undistort) + crop theo ROI để ảnh đẹp, không méo
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))  # alpha=0 để loại bỏ viền đen
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w_roi, h_roi = roi
    undistorted_cropped = undistorted[y:y + h_roi, x:x + w_roi]

    if frame_index % SKIP_FRAME == 0:
        start_infer = time.time()
        results = model(undistorted_cropped, verbose=False)[0]
        infer_time = time.time() - start_infer

        yolo_infer_times.append(infer_time)
        if len(yolo_infer_times) > 30:
            yolo_infer_times.pop(0)
        avg_infer_time = sum(yolo_infer_times) / len(yolo_infer_times)
        yolo_fps = 1.0 / avg_infer_time if avg_infer_time > 0 else 0

        # Vẽ bounding box + khoảng cách
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = model.names[cls_id]
            pixel_height = y2 - y1

            if cls_id in REAL_HEIGHTS and pixel_height > 0:
                real_height = REAL_HEIGHTS[cls_id]
                distance = ALPHA * (real_height * focal_length) / pixel_height

                cv2.rectangle(undistorted_cropped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    undistorted_cropped,
                    f"{label} {distance:.2f}m",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
        annotated_frame = undistorted_cropped.copy()

    elif annotated_frame is None:
        annotated_frame = undistorted_cropped.copy()

    # Tính FPS tổng thể
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Hiển thị FPS
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"YOLO FPS: {yolo_fps:.2f}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Hiển thị hình ảnh đã hiệu chỉnh và crop
    cv2.imshow("YOLOv8 + Khoảng cách (Undistorted + Cropped)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
