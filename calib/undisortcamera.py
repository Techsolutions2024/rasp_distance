import cv2
import numpy as np

# Load thông số hiệu chỉnh
data = np.load("camera_calib.npz")
mtx = data["camera_matrix"]
dist = data["dist_coeffs"]

# Khởi tạo webcam (đổi số nếu cần)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Không mở được webcam.")
    exit()

print("🎥 Đang chạy undistort live. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không đọc được khung hình.")
        break

    h, w = frame.shape[:2]

    # Tính toán ma trận camera mới
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Undistort khung hình
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Cắt vùng ROI nếu cần (bỏ nếu muốn xem toàn bộ)
    x, y, w, h = roi
    undistorted_cropped = undistorted[y:y+h, x:x+w]

    # Hiển thị song song
    combined = np.hstack((cv2.resize(frame, (w, h)), cv2.resize(undistorted_cropped, (w, h))))
    cv2.imshow("Goc trai: Goc | Goc phai: Sau undistort", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
