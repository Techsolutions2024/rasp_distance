import cv2
import numpy as np

# Load thông số hiệu chỉnh từ file
data = np.load("camera_calib.npz")
mtx = data["camera_matrix"]
dist = data["dist_coeffs"]

# Đọc ảnh cần kiểm tra
img = cv2.imread("test.jpg")  # 👈 đổi tên nếu bạn có ảnh khác

if img is None:
    print("Không đọc được ảnh 'test.jpg'. Hãy đặt ảnh vào cùng thư mục.")
    exit()

h, w = img.shape[:2]

# Tính toán ma trận mới để undistort
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Undistort ảnh
undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Cắt vùng ROI hợp lệ (nếu muốn)
x, y, w, h = roi
undistorted_cropped = undistorted[y:y+h, x:x+w]

# Hiển thị ảnh gốc và ảnh đã chỉnh méo
cv2.imshow("Ảnh Gốc", img)
cv2.imshow("Ảnh Sau Undistort", undistorted_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
