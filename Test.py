import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
from collections import deque
import queue

class AccurateDetectionPipeline:
    def __init__(self, model_path="best_int8_openvino_model/", camera_id=0):
        # Cấu hình để có bbox chính xác
        self.CAMERA_RESOLUTION = (640, 480)  # Resolution gốc từ camera
        self.DETECTION_RESOLUTION = (416, 320)  # Resolution cho YOLO
        self.SKIP_FRAMES = 0  # Không skip frame để tracking liên tục
        self.BUFFER_SIZE = 2  # Tăng buffer để không drop frame
        self.ALPHA = 0.8
        
        # Khởi tạo camera calibration
        self._init_camera_calibration()
        
        # Khởi tạo YOLO model
        self._init_yolo_model(model_path)
        
        # Queues với buffer lớn hơn
        self.frame_queue = queue.Queue(maxsize=3)
        self.detection_queue = queue.Queue(maxsize=2)
        
        # Control flags
        self.running = True
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.detection_fps = 0
        
        # Pre-allocated buffers
        self.detection_buffer = np.zeros((*self.DETECTION_RESOLUTION[::-1], 3), dtype=np.uint8)
        
        self.camera_id = camera_id
        
        # ============ CẤU HÌNH DETECTION ============
        
        # OPTION 1: CHỈ detect một số class nhất định
        # Nếu muốn CHỈ detect những class này, uncomment dòng dưới:
        self.ALLOWED_CLASSES = {4,8,9, 3, 6,5, 7}  # chỉ detect person, car, bus, truck
        #self.ALLOWED_CLASSES = None  # None = detect tất cả class
        
        # OPTION 2: Các class sẽ tính khoảng cách (có thể khác với ALLOWED_CLASSES)
        self.CUSTOM_REAL_HEIGHTS = {
            #0: 1.7,   # person - chiều cao người (m)
            #1: 1.6,   # bicycle
            #2: 4.0,   # 
            3: 3.2,   # bus
            4: 1.6,   # xe đạp
            5: 4.2,   # xe đầu kéo
            6: 3.2,   # xe du lịch 
            7: 1.5,   # xe máy
            8: 2.3,   # ô tô
            9: 3.0,   # xe tải
            # Thêm các class khác theo nhu cầu
            # 10: 2.0,  # fire hydrant
            # 11: 1.2,  # stop sign
            # ...
        }
        
        # ==========================================
        
        # Tính toán scale factors chính xác
        self.scale_x = self.CAMERA_RESOLUTION[0] / self.DETECTION_RESOLUTION[0]
        self.scale_y = self.CAMERA_RESOLUTION[1] / self.DETECTION_RESOLUTION[1]
        
    def _init_camera_calibration(self):
        """Khởi tạo camera calibration với tọa độ chính xác"""
        try:
            data = np.load("camera_calib.npz")
            self.mtx = data["camera_matrix"]
            self.dist = data["dist_coeffs"]
            
            # Tạo undistort map cho camera resolution
            self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, self.CAMERA_RESOLUTION, 0.3, self.CAMERA_RESOLUTION
            )
            
            # Map cho camera resolution
            self.camera_map1, self.camera_map2 = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, self.newcameramtx, 
                self.CAMERA_RESOLUTION, cv2.CV_16SC2
            )
            
            # Tính focal length
            f_x = self.newcameramtx[0, 0]
            f_y = self.newcameramtx[1, 1]
            self.focal_length = (f_x + f_y) / 2
            
            # Tính ROI offset để điều chỉnh tọa độ
            self.roi_x_offset = self.roi[0]
            self.roi_y_offset = self.roi[1]
            
            self.use_calibration = True
            print(f"✓ Camera calibration loaded - ROI: {self.roi}")
            
        except Exception as e:
            print(f"⚠ Using default calibration: {e}")
            self.use_calibration = False
            self.focal_length = 350
            self.roi = (0, 0, self.CAMERA_RESOLUTION[0], self.CAMERA_RESOLUTION[1])
            self.roi_x_offset = 0
            self.roi_y_offset = 0
    
    def _init_yolo_model(self, model_path):
        """Khởi tạo YOLO model"""
        print("🔄 Loading YOLO model...")
        self.model = YOLO(model_path)
        
        # Cấu hình YOLO
        self.model.overrides.update({
            'verbose': False,
            'conf': 0.3,      # Giảm confidence để detect nhiều hơn
            'iou': 0.7,       
            'max_det': 20,    
            'device': 'cpu',
            'half': False,
            'agnostic_nms': False,
            'retina_masks': False,
        })
        
        print("✓ YOLO model loaded")
    
    def camera_thread(self):
        """Thread đọc camera - không drop frame"""
        cap = cv2.VideoCapture(self.camera_id)
        
        # Cấu hình camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_RESOLUTION[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            self.running = False
            return
        
        print(f"✓ Camera ready at {self.CAMERA_RESOLUTION}")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Undistort frame gốc nếu cần
            if self.use_calibration:
                undistorted_frame = cv2.remap(frame, self.camera_map1, self.camera_map2, cv2.INTER_LINEAR)
                # Crop theo ROI
                x, y, w, h = self.roi
                processed_frame = undistorted_frame[y:y+h, x:x+w]
            else:
                processed_frame = frame
            
            # Put frame (blocking nhưng với timeout)
            try:
                self.frame_queue.put(processed_frame, timeout=0.01)
            except queue.Full:
                # Nếu queue đầy, lấy frame cũ ra và put frame mới
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(processed_frame)
                except queue.Empty:
                    pass
        
        cap.release()
        print("📷 Camera stopped")
    
    def detection_thread(self):
        """Thread detection - xử lý mọi frame"""
        while self.running:
            try:
                # Lấy frame
                frame = self.frame_queue.get(timeout=0.05)
                
                # Resize cho detection (giữ tỷ lệ)
                detection_frame = cv2.resize(frame, self.DETECTION_RESOLUTION)
                
                # YOLO inference
                start_time = time.time()
                results = self.model(detection_frame, verbose=False)[0]
                inference_time = time.time() - start_time
                
                # Tính detection FPS
                if inference_time > 0:
                    self.detection_fps = 1.0 / inference_time
                
                # Xử lý detections với tọa độ chính xác
                detections = self._process_detections_accurate(results, frame.shape)
                
                # Gửi kết quả
                detection_data = {
                    'frame': frame,  # Frame gốc đã undistort
                    'detections': detections,
                    'timestamp': time.time()
                }
                
                # Put detection (non-blocking)
                try:
                    # Nếu queue đầy, drop detection cũ
                    if self.detection_queue.full():
                        try:
                            self.detection_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.detection_queue.put_nowait(detection_data)
                except queue.Full:
                    pass
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection error: {e}")
                continue
        
        print("🎯 Detection stopped")
    
    def _process_detections_accurate(self, results, original_shape):
        """Xử lý detections với tọa độ chính xác"""
        detections = []
        
        if results.boxes is None or len(results.boxes) == 0:
            return detections
        
        # Tính scale factor chính xác
        original_h, original_w = original_shape[:2]
        scale_x = original_w / self.DETECTION_RESOLUTION[0]
        scale_y = original_h / self.DETECTION_RESOLUTION[1]
        
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            if conf < 0.4:  # Threshold thấp để detect nhiều
                continue
            
            # LỌC CLASS: Chỉ giữ lại class được phép (nếu có set ALLOWED_CLASSES)
            if self.ALLOWED_CLASSES is not None and cls_id not in self.ALLOWED_CLASSES:
                continue  # Skip class này
            
            # Chuyển đổi tọa độ chính xác
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Scale về kích thước frame gốc
            x1_scaled = int(x1 * scale_x)
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x)
            y2_scaled = int(y2 * scale_y)
            
            # Đảm bảo tọa độ trong bounds
            x1_scaled = max(0, min(x1_scaled, original_w - 1))
            y1_scaled = max(0, min(y1_scaled, original_h - 1))
            x2_scaled = max(x1_scaled + 1, min(x2_scaled, original_w))
            y2_scaled = max(y1_scaled + 1, min(y2_scaled, original_h))
            
            # Tính khoảng cách cho các class được chỉ định
            distance = None
            if cls_id in self.CUSTOM_REAL_HEIGHTS:
                pixel_height = y2_scaled - y1_scaled
                if pixel_height > 10:
                    real_height = self.CUSTOM_REAL_HEIGHTS[cls_id]
                    # Sử dụng focal length đã calibration
                    distance = self.ALPHA * (real_height * self.focal_length) / pixel_height
            
            detections.append({
                'bbox': (x1_scaled, y1_scaled, x2_scaled, y2_scaled),
                'class_id': cls_id,
                'confidence': conf,
                'label': cls_id,  # ✅ THAY ĐỔI: Sử dụng class_id thay vì tên
                'distance': distance
            })
        
        return detections
    
    def display_thread(self):
        """Thread hiển thị"""
        while self.running:
            try:
                detection_data = self.detection_queue.get(timeout=0.1)
                
                frame = detection_data['frame'].copy()
                detections = detection_data['detections']
                
                # Vẽ detections
                self._draw_detections(frame, detections)
                
                # Tính FPS
                self.fps_counter += 1
                current_time = time.time()
                
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                
                # Vẽ thông tin
                self._draw_info(frame)
                
                # Hiển thị
                cv2.imshow("Accurate YOLO Detection", frame)
                
                # Check quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                
            except queue.Empty:
                continue
        
        cv2.destroyAllWindows()
        print("🖥️ Display stopped")
    
    def _draw_detections(self, frame, detections):
        """Vẽ detections với thông tin đầy đủ - HIỂN THỊ ID thay vì tên"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']  # ✅ Lấy class_id
            conf = det['confidence']
            distance = det['distance']
            
            # Màu sắc theo class
            color = self._get_class_color(class_id)
            
            # Vẽ bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # ✅ THAY ĐỔI: Hiển thị ID thay vì tên
            if distance is not None:
                text = f"ID:{class_id} {conf:.2f} - {distance:.1f}m"
            else:
                text = f"ID:{class_id} {conf:.2f}"
            
            # Background cho text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0], y1), color, -1)
            
            # Text
            cv2.putText(frame, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _get_class_color(self, class_id):
        """Lấy màu cho từng class với mapping chính xác"""
        # Mapping màu theo class_id cụ thể
        color_map = {
            7: (255, 0, 255),    # ID:0 - magenta
            3: (0, 255, 0),      # ID:3 - green
            4: (255, 0, 0),      # ID:4 - blue  
            5: (0, 0, 255),      # ID:5 - red
            6: (0, 255, 0),    # ID:6 - cyan/yellow
            8: (0, 255, 0),    # ID:8 - yellow
            9: (128, 0, 128),    # ID:9 - purple
        }
        
        # Trả về màu mặc định nếu class_id không có trong map
        return color_map.get(class_id, (128, 128, 128))  # Gray mặc định
    
    def _draw_info(self, frame):
        """Vẽ thông tin FPS và status"""
        # FPS
        cv2.putText(frame, f"FPS: {self.detection_fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def run(self):
        """Chạy pipeline"""
        print("🚀 Starting accurate detection pipeline...")
        print(f"📐 Camera: {self.CAMERA_RESOLUTION}")
        print(f"🎯 Detection: {self.DETECTION_RESOLUTION}")
        print(f"📏 Scale factors: {self.scale_x:.2f}x, {self.scale_y:.2f}x")
        
        if self.ALLOWED_CLASSES is not None:
            print(f"🎪 Only detect class IDs: {sorted(list(self.ALLOWED_CLASSES))}")
        else:
            print(f"🎪 Detect ALL classes, distance for: {list(self.CUSTOM_REAL_HEIGHTS.keys())}")
        
        print(f"📏 Distance calculation for class IDs: {list(self.CUSTOM_REAL_HEIGHTS.keys())}")
        
        # Khởi động threads
        threads = [
            threading.Thread(target=self.camera_thread, name="Camera"),
            threading.Thread(target=self.detection_thread, name="Detection"),
            threading.Thread(target=self.display_thread, name="Display")
        ]
        
        for thread in threads:
            thread.daemon = True
            thread.start()
        
        try:
            threads[2].join()  # Chờ display thread
        except KeyboardInterrupt:
            print("\n⏹️ Stopping...")
            self.running = False
        
        # Cleanup
        for thread in threads[:-1]:
            thread.join(timeout=1.0)
        
        print("✅ Pipeline stopped")

# Sử dụng
if __name__ == "__main__":
    pipeline = AccurateDetectionPipeline(
        model_path="best_int8_openvino_model/",
        camera_id=0
    )
    pipeline.run()
