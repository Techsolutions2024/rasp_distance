import cv2
import numpy as np
import glob
import os

# Kích thước checkerboard: số điểm góc bên trong
CHECKERBOARD = (7, 6)

# Kích thước ô (mm hoặc cm, tùy bạn đo thực tế), ví dụ: mỗi ô là 25mm
square_size = 25.0

# Đường dẫn tới ảnh checkerboard
image_dir = "calib_images"
images = glob.glob(os.path.join(image_dir, "*.jpg"))

# Danh sách để lưu điểm 3D và 2D
objpoints = []  # Điểm 3D thực tế (trên mặt phẳng checkerboard)
imgpoints = []  # Điểm 2D phát hiện trên ảnh

# Tạo mảng điểm object (giả sử z=0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size  # Nhân với kích thước ô thực tế

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        # Vẽ góc và hiển thị
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("Detected", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Lấy kích thước ảnh cuối cùng
h, w = gray.shape[:2]

# Hiệu chỉnh camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (w, h), None, None
)

# In kết quả
print("\n🎯 Calibration Results:")
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist.ravel())

# Lưu ra file
np.savez("camera_calib.npz", camera_matrix=mtx, dist_coeffs=dist)
print("✅ Đã lưu file: camera_calib.npz")
