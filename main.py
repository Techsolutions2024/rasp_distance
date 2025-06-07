import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
from collections import deque
import queue

class LightweightDetectionPipeline:
    def __init__(self, model_path="best_int8_openvino_model/", camera_id=1):
        # Cấu hình tối ưu
        self.RESOLUTION = (416, 320)
        self.DISPLAY_RESOLUTION = (640, 480)
        self.SKIP_FRAMES = 4  # Tăng skip để giảm tải
        self.BUFFER_SIZE = 1  # Giảm buffer để giảm latency
        self.ALPHA = 0.8
        
        # Khởi tạo camera calibration
        self._init_camera_calibration()
        
        # Khởi tạo YOLO model
        self._init_yolo_model(model_path)
        
        # Lightweight queues
        self.frame_queue = queue.Queue(maxsize=1)
        self.detection_queue = queue.Queue(maxsize=1)
        
        # Control flags
        self.running = True
        
        # LIGHTWEIGHT FPS tracking - chỉ track cái cần thiết
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.last_fps_update = 0
        
        # Detection timing (chỉ track inference time)
        self.last_inference_time = 0
        self.detection_fps = 0
        
        # Pre-allocated buffers
        self.processing_buffer = np.zeros((*self.RESOLUTION[::-1], 3), dtype=np.uint8)
        
        self.camera_id = camera_id
        
    def _init_camera_calibration(self):
        """Khởi tạo camera calibration - tối ưu hóa"""
        try:
            data = np.load("camera_calib.npz")
            self.mtx = data["camera_matrix"]
            self.dist = data["dist_coeffs"]
            
            # Chỉ tạo map cho detection resolution (không cần display resolution)
            self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, self.RESOLUTION, 0.3, self.RESOLUTION  # alpha=0.3 thay vì 0
            )
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, self.newcameramtx, 
                self.RESOLUTION, cv2.CV_16SC2
            )
            
            # Tính focal length
            f_x = self.newcameramtx[0, 0]
            f_y = self.newcameramtx[1, 1]
            self.focal_length = (f_x + f_y) / 2
            
            self.use_calibration = True
            print(f"✓ Camera calibration loaded")
            
        except Exception as e:
            print(f"⚠ Using default calibration: {e}")
            self.use_calibration = False
            self.focal_length = 350
            self.roi = (0, 0, self.RESOLUTION[0], self.RESOLUTION[1])
    
    def _init_yolo_model(self, model_path):
        """Khởi tạo YOLO model với cấu hình siêu tối ưu"""
        print("🔄 Loading YOLO model...")
        self.model = YOLO(model_path)
        
        # Cấu hình siêu tối ưu cho Pi 4
        self.model.overrides.update({
            'verbose': False,
            'conf': 0.6,      # Tăng confidence để giảm post-processing
            'iou': 0.8,       # Tăng IoU để giảm NMS
            'max_det': 15,    # Giảm max detection
            'device': 'cpu',
            'half': False,
            'agnostic_nms': False,  # Tắt agnostic NMS
            'retina_masks': False,  # Tắt retina masks
        })
        
        # Simplified real heights
        self.REAL_HEIGHTS = {
            0: 1.6, 1: 1.6, 2: 4.0, 3: 3.2, 4: 1.1,
            5: 4.2, 6: 3.2, 7: 1.5, 8: 1.5, 9: 3.0
        }
        
        print("✓ YOLO model loaded")
    
    def camera_thread(self):
        """Thread đọc camera - đơn giản hóa tối đa"""
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        # Cấu hình camera tối ưu
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.DISPLAY_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.DISPLAY_RESOLUTION[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            self.running = False
            return
        
        print(f"✓ Camera ready")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Resize ngay khi đọc để giảm memory
            frame_small = cv2.resize(frame, self.RESOLUTION)
            
            # Chỉ put frame nếu queue trống (drop old frames)
            if self.frame_queue.empty():
                try:
                    self.frame_queue.put_nowait(frame_small)
                except queue.Full:
                    pass
        
        cap.release()
        print("📷 Camera stopped")
    
    def detection_thread(self):
        """Thread detection - tối ưu hóa tối đa"""
        frame_counter = 0
        
        while self.running:
            try:
                # Lấy frame (timeout ngắn)
                frame = self.frame_queue.get(timeout=0.05)
                frame_counter += 1
                
                # Skip frames để giảm tải
                if frame_counter % self.SKIP_FRAMES != 0:
                    continue
                
                # Undistort nếu cần (chỉ cho detection)
                if self.use_calibration:
                    cv2.remap(frame, self.map1, self.map2, 
                             cv2.INTER_LINEAR, self.processing_buffer)
                    x, y, w, h = self.roi
                    detection_frame = self.processing_buffer[y:y+h, x:x+w]
                else:
                    detection_frame = frame
                
                # YOLO inference
                start_time = time.time()
                results = self.model(detection_frame, verbose=False)[0]
                self.last_inference_time = time.time() - start_time
                
                # Tính detection FPS đơn giản
                if self.last_inference_time > 0:
                    self.detection_fps = 1.0 / self.last_inference_time
                
                # Xử lý kết quả nhanh
                detections = self._process_detections_fast(results)
                
                # Resize frame về display size để hiển thị
                display_frame = cv2.resize(frame, self.DISPLAY_RESOLUTION)
                
                # Gửi kết quả (non-blocking)
                detection_data = {
                    'frame': display_frame,
                    'detections': detections,
                    'timestamp': time.time()
                }
                
                # Drop old detection nếu queue đầy
                if not self.detection_queue.empty():
                    try:
                        self.detection_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                try:
                    self.detection_queue.put_nowait(detection_data)
                except queue.Full:
                    pass
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection error: {e}")
                continue
        
        print("🎯 Detection stopped")
    
    def _process_detections_fast(self, results):
        """Xử lý detection nhanh - tối ưu hóa"""
        detections = []
        
        if results.boxes is None or len(results.boxes) == 0:
            return detections
        
        # Scale factor từ detection về display
        scale_x = self.DISPLAY_RESOLUTION[0] / self.RESOLUTION[0]
        scale_y = self.DISPLAY_RESOLUTION[1] / self.RESOLUTION[1]
        
        # Chỉ lấy top detections
        boxes = results.boxes[:10]  # Giới hạn để tăng tốc
        
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            if conf < 0.5:  # Skip low confidence
                continue
            
            # Scale coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            # Tính khoảng cách nhanh
            distance = None
            if cls_id in self.REAL_HEIGHTS:
                pixel_height = y2 - y1
                if pixel_height > 15:  # Tăng threshold
                    real_height = self.REAL_HEIGHTS[cls_id]
                    distance = self.ALPHA * (real_height * self.focal_length) / pixel_height
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'class_id': cls_id,
                'confidence': conf,
                'label': self.model.names[cls_id],
                'distance': distance
            })
        
        return detections
    
    def display_thread(self):
        """Thread hiển thị - tối ưu FPS tracking"""
        last_frame = None
        
        while self.running:
            try:
                # Lấy detection data
                detection_data = self.detection_queue.get(timeout=0.1)
                
                frame = detection_data['frame']
                detections = detection_data['detections']
                
                # Vẽ detections
                annotated_frame = self._draw_detections_fast(frame, detections)
                
                # LIGHTWEIGHT FPS tracking - chỉ tính tổng FPS
                self.fps_counter += 1
                current_time = time.time()
                
                # Cập nhật FPS mỗi giây (thay vì mỗi frame)
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                    self.last_fps_update = current_time
                
                # Vẽ FPS info (đơn giản)
                self._draw_simple_fps(annotated_frame, current_time)
                
                # Hiển thị
                cv2.imshow("YOLO Detection", annotated_frame)
                last_frame = annotated_frame
                
            except queue.Empty:
                # Hiển thị frame cũ nếu không có detection mới
                if last_frame is not None:
                    cv2.imshow("YOLO Detection", last_frame)
                continue
            
            # Check quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        
        cv2.destroyAllWindows()
        print("🖥️ Display stopped")
    
    def _draw_detections_fast(self, frame, detections):
        """Vẽ detections nhanh"""
        annotated_frame = frame.copy()
        
        for det in detections[:8]:  # Giới hạn số detections để vẽ
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            distance = det['distance']
            
            # Vẽ bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label đơn giản
            if distance is not None:
                text = f"{label} {distance:.1f}m"
            else:
                text = label
            
            cv2.putText(annotated_frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return annotated_frame
    
    def _draw_simple_fps(self, frame, current_time):
        """Vẽ FPS info đơn giản - chỉ hiển thị khi có update"""
        # Chỉ vẽ FPS nếu có update gần đây (trong 2 giây)
        if current_time - self.last_fps_update < 2.0:
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Hiển thị detection FPS nếu có
            if self.detection_fps > 0:
                cv2.putText(frame, f"Det: {self.detection_fps:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    def run(self):
        """Chạy pipeline tối ưu"""
        print("🚀 Starting lightweight pipeline...")
        
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
            thread.join(timeout=0.5)
        
        print("✅ Pipeline stopped")

# Sử dụng
if __name__ == "__main__":
    pipeline = LightweightDetectionPipeline(
        model_path="best_int8_openvino_model/",
        camera_id=1
    )
    pipeline.run()
