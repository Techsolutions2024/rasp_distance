import cv2
import os
import numpy as np

# Thông số checkerboard: số điểm góc bên trong
CHECKERBOARD = (7, 6)  # tương ứng bảng 8x7 ô

# Tạo thư mục lưu ảnh nếu chưa có
save_dir = "calib_images"
os.makedirs(save_dir, exist_ok=True)

# Khởi tạo webcam
cap = cv2.VideoCapture(1)  # 0 là webcam mặc định

if not cap.isOpened():
    print("Không mở được webcam.")
    exit()

print("Ấn 's' để lưu ảnh khi nhận được checkerboard. Ấn 'q' để thoát.")

img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được khung hình.")
        break

    # Resize nhỏ lại nếu cần (tùy máy yếu)
    # frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    display = frame.copy()

    if ret_cb:
        # Tô góc phát hiện được
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners, ret_cb)
        cv2.putText(display, "Checkerboard FOUND", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(display, "Checkerboard NOT found", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Capture Checkerboard", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and ret_cb:
        filename = os.path.join(save_dir, f"checkerboard_{img_counter:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Đã lưu ảnh: {filename}")
        img_counter += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
