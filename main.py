import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO

# --- Thông số hiệu chỉnh khoảng cách ---
ALPHA = 0.8

# Cấu hình tối ưu cho Pi 4
RESOLUTION = (416, 320)  # Giảm resolution để tăng tốc
SKIP_FRAME = 4  # Tăng skip frame (chỉ detect mỗi 4 frame)
MAX_BUFFER_SIZE = 2  # Giảm buffer

# Load thông số hiệu chỉnh camera (tối ưu hóa một lần)
try:
    data = np.load("camera_calib.npz")
    mtx = data["camera_matrix"]
    dist = data["dist_coeffs"]
    
    # Pre-compute undistortion map để tránh tính toán lặp lại
    h, w = RESOLUTION[1], RESOLUTION[0]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), cv2.CV_16SC2)
    use_undistort = True
    print("Sử dụng camera calibration")
except:
    use_undistort = False
    roi = (0, 0, RESOLUTION[0], RESOLUTION[1])
    print("Không tìm thấy camera calibration, bỏ qua undistortion")

# Tính tiêu cự (nếu có calibration)
if use_undistort:
    f_x = mtx[0, 0] * (RESOLUTION[0] / 640)  # Scale theo resolution mới
    f_y = mtx[1, 1] * (RESOLUTION[1] / 480)
    focal_length = (f_x + f_y) / 2
else:
    focal_length = 500  # Giá trị ước lượng

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

# Load mô hình YOLOv8 với cấu hình tối ưu
print("Đang load YOLO model...")
model = YOLO("best_int8_openvino_model/")
model.overrides['verbose'] = False
model.overrides['conf'] = 0.5  # Tăng confidence threshold để giảm số detection
model.overrides['iou'] = 0.7   # Tăng IoU threshold để giảm NMS

# Biến global tối ưu
frame_buffer = []
buffer_lock = threading.Lock()
detection_results = None
results_lock = threading.Lock()

# Pre-allocate arrays để tránh memory allocation
processed_frame = np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)

def camera_thread():
    """Thread đọc camera với buffer giới hạn"""
    global frame_buffer
    
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer của camera
    
    if not cap.isOpened():
        print("Không thể mở webcam")
        return
    
    print(f"Camera resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        with buffer_lock:
            if len(frame_buffer) >= MAX_BUFFER_SIZE:
                frame_buffer.pop(0)  # Xóa frame cũ
            frame_buffer.append(frame)
    
    cap.release()

def detection_thread():
    """Thread riêng cho YOLO detection"""
    global detection_results
    
    while True:
        current_frame = None
        
        # Lấy frame mới nhất
        with buffer_lock:
            if frame_buffer:
                current_frame = frame_buffer[-1].copy()
        
        if current_frame is None:
            time.sleep(0.01)
            continue
        
        # Xử lý frame
        if use_undistort:
            # Sử dụng remap thay vì undistort (nhanh hơn ~30%)
            cv2.remap(current_frame, map1, map2, cv2.INTER_LINEAR, processed_frame)
            x, y, w_roi, h_roi = roi
            processed_frame = processed_frame[y:y + h_roi, x:x + w_roi]
        else:
            processed_frame = current_frame
        
        # YOLO detection
        start_time = time.time()
        results = model(processed_frame, verbose=False)[0]
        detection_time = time.time() - start_time
        
        # Lưu kết quả
        with results_lock:
            detection_results = {
                'boxes': results.boxes,
                'frame': processed_frame.copy(),
                'detection_time': detection_time,
                'timestamp': time.time()
            }
        
        # Tạm dừng để tránh CPU 100%
        time.sleep(0.05)

def main():
    global detection_results
    
    # Khởi động threads
    cam_thread = threading.Thread(target=camera_thread)
    cam_thread.daemon = True
    cam_thread.start()
    
    det_thread = threading.Thread(target=detection_thread)
    det_thread.daemon = True
    det_thread.start()
    
    # Đợi camera khởi động
    time.sleep(2)
    
    # Biến FPS
    frame_count = 0
    fps_start_time = time.time()
    display_fps = 0
    yolo_fps = 0
    
    last_detection_time = 0
    
    print("Bắt đầu detection. Nhấn 'q' để thoát.")
    
    while True:
        display_frame = None
        current_detection = None
        
        # Lấy frame hiện tại từ buffer
        with buffer_lock:
            if frame_buffer:
                display_frame = frame_buffer[-1].copy()
        
        # Lấy kết quả detection mới nhất
        with results_lock:
            if detection_results and detection_results['timestamp'] > last_detection_time:
                current_detection = detection_results.copy()
                last_detection_time = current_detection['timestamp']
        
        if display_frame is None:
            continue
        
        # Xử lý frame để hiển thị (đơn giản hóa)
        if use_undistort:
            cv2.remap(display_frame, map1, map2, cv2.INTER_LINEAR, processed_frame)
            x, y, w_roi, h_roi = roi
            display_frame = processed_frame[y:y + h_roi, x:x + w_roi]
        
        # Vẽ detection results nếu có
        if current_detection and current_detection['boxes'] is not None:
            yolo_fps = 1.0 / current_detection['detection_time'] if current_detection['detection_time'] > 0 else 0
            
            for box in current_detection['boxes']:
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                
                if conf < 0.5:  # Skip low confidence
                    continue
                
                label = model.names[cls_id]
                pixel_height = y2 - y1
                
                # Tính khoảng cách (nếu có trong REAL_HEIGHTS)
                distance_text = ""
                if cls_id in REAL_HEIGHTS and pixel_height > 10:  # Tránh divide by zero
                    real_height = REAL_HEIGHTS[cls_id]
                    distance = ALPHA * (real_height * focal_length) / pixel_height
                    distance_text = f" ({distance:.1f}m)"
                
                # Vẽ bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Vẽ label (tối ưu hóa text)
                text = f"{label}{distance_text}"
                cv2.putText(display_frame, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Tính FPS display
        frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            display_fps = frame_count / (time.time() - fps_start_time)
            frame_count = 0
            fps_start_time = time.time()
        
        # Hiển thị FPS (đơn giản hóa)
        cv2.putText(display_frame, f"FPS: {display_fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"YOLO: {yolo_fps:.1f}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Hiển thị
        cv2.imshow("Optimized YOLO", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
