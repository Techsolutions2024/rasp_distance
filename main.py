import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
from collections import deque
import queue

class OptimizedDetectionPipeline:
    def __init__(self, model_path="best_int8_openvino_model/", camera_id=1):
        # Cấu hình tối ưu
        self.RESOLUTION = (416, 320)  # Giảm resolution để tăng tốc
        self.DISPLAY_RESOLUTION = (640, 480)  # Resolution hiển thị
        self.SKIP_FRAMES = 3  # Detect mỗi 3 frame
        self.BUFFER_SIZE = 2
        self.ALPHA = 0.8
        
        # Khởi tạo camera calibration
        self._init_camera_calibration()
        
        # Khởi tạo YOLO model
        self._init_yolo_model(model_path)
        
        # Threading và queues
        self.frame_queue = queue.Queue(maxsize=self.BUFFER_SIZE)
        self.detection_queue = queue.Queue(maxsize=1)
        self.display_queue = queue.Queue(maxsize=1)
        
        # Control flags
        self.running = True
        self.stats = {
            'camera_fps': 0,
            'detection_fps': 0,
            'display_fps': 0,
            'total_fps': 0
        }
        
        # Pre-allocated buffers để tránh memory allocation
        self.processing_buffer = np.zeros((*self.RESOLUTION[::-1], 3), dtype=np.uint8)
        self.display_buffer = np.zeros((*self.DISPLAY_RESOLUTION[::-1], 3), dtype=np.uint8)
        
        # Camera setup
        self.camera_id = camera_id
        
    def _init_camera_calibration(self):
        """Khởi tạo camera calibration một lần"""
        try:
            data = np.load("camera_calib.npz")
            self.mtx = data["camera_matrix"]
            self.dist = data["dist_coeffs"]
            
            # Pre-compute undistortion maps cho cả 2 resolution
            # Processing resolution (cho YOLO)
            self.newcameramtx_proc, self.roi_proc = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, self.RESOLUTION, 0, self.RESOLUTION
            )
            self.map1_proc, self.map2_proc = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, self.newcameramtx_proc, 
                self.RESOLUTION, cv2.CV_16SC2
            )
            
            # Display resolution (cho hiển thị)
            self.newcameramtx_disp, self.roi_disp = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, self.DISPLAY_RESOLUTION, 0, self.DISPLAY_RESOLUTION
            )
            self.map1_disp, self.map2_disp = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, self.newcameramtx_disp, 
                self.DISPLAY_RESOLUTION, cv2.CV_16SC2
            )
            
            # Tính focal length cho processing resolution
            f_x = self.newcameramtx_proc[0, 0]
            f_y = self.newcameramtx_proc[1, 1]
            self.focal_length = (f_x + f_y) / 2
            
            self.use_calibration = True
            print(f"✓ Camera calibration loaded - Processing: {self.RESOLUTION}, Display: {self.DISPLAY_RESOLUTION}")
            
        except Exception as e:
            print(f"⚠ Camera calibration failed: {e}")
            self.use_calibration = False
            self.focal_length = 400  # Estimated focal length
            self.roi_proc = (0, 0, self.RESOLUTION[0], self.RESOLUTION[1])
            self.roi_disp = (0, 0, self.DISPLAY_RESOLUTION[0], self.DISPLAY_RESOLUTION[1])
    
    def _init_yolo_model(self, model_path):
        """Khởi tạo YOLO model với cấu hình tối ưu"""
        print("🔄 Loading YOLO model...")
        self.model = YOLO(model_path)
        
        # Tối ưu cấu hình YOLO
        self.model.overrides.update({
            'verbose': False,
            'conf': 0.5,      # Tăng confidence để giảm false positive
            'iou': 0.7,       # Tăng IoU để giảm NMS computation
            'max_det': 20,    # Giới hạn số detection
            'device': 'cpu',
            'half': False,    # Tắt FP16 trên CPU
        })
        
        # Real heights dictionary
        self.REAL_HEIGHTS = {
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
        
        print("✓ YOLO model loaded successfully")
    
    def camera_thread(self):
        """Thread 1: Đọc frame từ camera"""
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        # Cấu hình camera tối ưu
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.DISPLAY_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.DISPLAY_RESOLUTION[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer lag
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            self.running = False
            return
        
        print(f"✓ Camera opened: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        
        frame_count = 0
        fps_start = time.time()
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Chỉ undistort nếu cần thiết và queue không đầy
            if not self.frame_queue.full():
                if self.use_calibration:
                    # Undistort cho display resolution
                    cv2.remap(frame, self.map1_disp, self.map2_disp, 
                             cv2.INTER_LINEAR, self.display_buffer)
                    x, y, w, h = self.roi_disp
                    processed_frame = self.display_buffer[y:y+h, x:x+w].copy()
                else:
                    processed_frame = frame.copy()
                
                try:
                    self.frame_queue.put_nowait({
                        'frame': processed_frame,
                        'timestamp': time.time()
                    })
                except queue.Full:
                    pass  # Skip frame nếu queue đầy
            
            # Tính camera FPS
            frame_count += 1
            if time.time() - fps_start >= 1.0:
                self.stats['camera_fps'] = frame_count / (time.time() - fps_start)
                frame_count = 0
                fps_start = time.time()
        
        cap.release()
        print("📷 Camera thread stopped")
    
    def detection_thread(self):
        """Thread 2: YOLO detection"""
        frame_counter = 0
        detection_count = 0
        fps_start = time.time()
        
        while self.running:
            try:
                # Lấy frame mới nhất từ queue
                frame_data = self.frame_queue.get(timeout=0.1)
                frame_counter += 1
                
                # Chỉ detect mỗi SKIP_FRAMES frame
                if frame_counter % self.SKIP_FRAMES != 0:
                    continue
                
                display_frame = frame_data['frame']
                
                # Resize frame cho YOLO (nhỏ hơn để tăng tốc)
                detection_frame = cv2.resize(display_frame, self.RESOLUTION)
                
                # Undistort cho detection nếu cần
                if self.use_calibration:
                    cv2.remap(detection_frame, self.map1_proc, self.map2_proc,
                             cv2.INTER_LINEAR, self.processing_buffer)
                    x, y, w, h = self.roi_proc
                    detection_frame = self.processing_buffer[y:y+h, x:x+w]
                
                # YOLO inference
                start_time = time.time()
                results = self.model(detection_frame, verbose=False)[0]
                inference_time = time.time() - start_time
                
                # Xử lý kết quả detection
                detections = self._process_detections(results, display_frame.shape)
                
                # Gửi kết quả tới display thread
                detection_data = {
                    'frame': display_frame,
                    'detections': detections,
                    'inference_time': inference_time,
                    'timestamp': time.time()
                }
                
                # Chỉ giữ detection mới nhất
                if not self.detection_queue.empty():
                    try:
                        self.detection_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.detection_queue.put_nowait(detection_data)
                
                detection_count += 1
                if time.time() - fps_start >= 1.0:
                    self.stats['detection_fps'] = detection_count / (time.time() - fps_start)
                    detection_count = 0
                    fps_start = time.time()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection error: {e}")
                continue
        
        print("🎯 Detection thread stopped")
    
    def _process_detections(self, results, frame_shape):
        """Xử lý kết quả detection và scale về frame gốc"""
        detections = []
        
        if results.boxes is None:
            return detections
        
        # Scale factor từ detection frame về display frame
        scale_x = frame_shape[1] / self.RESOLUTION[0]
        scale_y = frame_shape[0] / self.RESOLUTION[1]
        
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            if conf < 0.4:  # Filter low confidence
                continue
            
            # Scale coordinates về display frame
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            # Tính khoảng cách
            distance = None
            if cls_id in self.REAL_HEIGHTS:
                pixel_height = y2 - y1
                if pixel_height > 10:  # Tránh division by zero
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
        """Thread 3: Hiển thị kết quả"""
        display_count = 0
        fps_start = time.time()
        last_frame = None
        
        while self.running:
            try:
                # Lấy detection mới nhất
                detection_data = self.detection_queue.get(timeout=0.1)
                
                frame = detection_data['frame']
                detections = detection_data['detections']
                
                # Vẽ detection results
                annotated_frame = self._draw_detections(frame, detections)
                
                # Vẽ FPS info
                self._draw_fps_info(annotated_frame)
                
                # Hiển thị
                cv2.imshow("Optimized YOLO Detection", annotated_frame)
                last_frame = annotated_frame
                
                display_count += 1
                if time.time() - fps_start >= 1.0:
                    self.stats['display_fps'] = display_count / (time.time() - fps_start)
                    self.stats['total_fps'] = display_count / (time.time() - fps_start)
                    display_count = 0
                    fps_start = time.time()
                
            except queue.Empty:
                # Hiển thị frame cuối nếu không có detection mới
                if last_frame is not None:
                    self._draw_fps_info(last_frame)
                    cv2.imshow("Optimized YOLO Detection", last_frame)
                continue
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        
        cv2.destroyAllWindows()
        print("🖥️ Display thread stopped")
    
    def _draw_detections(self, frame, detections):
        """Vẽ bounding boxes và labels"""
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            conf = det['confidence']
            distance = det['distance']
            
            # Vẽ bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Tạo label text
            if distance is not None:
                text = f"{label} {distance:.1f}m ({conf:.2f})"
            else:
                text = f"{label} ({conf:.2f})"
            
            # Vẽ label
            cv2.putText(annotated_frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return annotated_frame
    
    def _draw_fps_info(self, frame):
        """Vẽ thông tin FPS"""
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        fps_texts = [
            f"Total FPS: {self.stats['total_fps']:.1f}",
            f"Camera: {self.stats['camera_fps']:.1f}",
            f"Detection: {self.stats['detection_fps']:.1f}",
            f"Display: {self.stats['display_fps']:.1f}"
        ]
        
        for i, text in enumerate(fps_texts):
            cv2.putText(frame, text, (10, y_offset + i * 25),
                       font, 0.6, (0, 255, 0), 2)
    
    def run(self):
        """Chạy pipeline"""
        print("🚀 Starting optimized detection pipeline...")
        
        # Khởi động các threads
        threads = [
            threading.Thread(target=self.camera_thread, name="Camera"),
            threading.Thread(target=self.detection_thread, name="Detection"),
            threading.Thread(target=self.display_thread, name="Display")
        ]
        
        for thread in threads:
            thread.daemon = True
            thread.start()
        
        try:
            # Chờ display thread (chính)
            threads[2].join()
        except KeyboardInterrupt:
            print("\n⏹️ Stopping pipeline...")
            self.running = False
        
        # Chờ các threads khác dừng
        for thread in threads[:-1]:
            thread.join(timeout=1.0)
        
        print("✅ Pipeline stopped successfully")

# Sử dụng
if __name__ == "__main__":
    # Khởi tạo và chạy pipeline
    pipeline = OptimizedDetectionPipeline(
        model_path="best_int8_openvino_model/",
        camera_id=1
    )
    
    pipeline.run()
