import cv2
import numpy as np

f_x = 219.55166549
f_y = 219.23781574
focal_length = (f_x + f_y) / 2

samples = []
click_points = []

def mouse_callback(event, x, y, flags, param):
    global click_points

    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        print(f"📍 Đã chọn điểm {len(click_points)}: ({x}, {y})")

        if len(click_points) == 2:
            y1 = click_points[0][1]
            y2 = click_points[1][1]
            pixel_height = abs(y2 - y1)

            print(f"\n🎯 Chiều cao bounding box (pixel): {pixel_height}")

            # Nhập chiều cao thực và khoảng cách thật
            try:
                real_height = float(input("👉 Nhập chiều cao thực (m): "))
                true_distance = float(input("📏 Nhập khoảng cách thật từ camera (m): "))
            except ValueError:
                print("❌ Dữ liệu nhập sai, bỏ mẫu này.")
                click_points = []
                return

            # Tính alpha
            if pixel_height > 0:
                est_distance = (real_height * focal_length) / pixel_height
                alpha = true_distance / est_distance

                samples.append((pixel_height, real_height, true_distance, alpha))
                print(f"✅ Đã lưu mẫu: (px_height={pixel_height}, real={real_height}, true_dist={true_distance}, alpha={alpha:.4f})\n")
            else:
                print("❌ Bounding box không hợp lệ.")

            click_points = []  # Reset để chọn mẫu mới

def main():
    global samples

    # Đọc ảnh từ file (ảnh từ webcam hoặc YOLO)
    img_path = "checkerboard_00.jpg"  # đổi đường dẫn nếu cần
    img = cv2.imread(img_path)

    if img is None:
        print("❌ Không đọc được ảnh.")
        return

    cv2.namedWindow("Chọn 2 điểm: đỉnh & chân vật thể")
    cv2.setMouseCallback("Chọn 2 điểm: đỉnh & chân vật thể", mouse_callback)

    print("🖱 Click chuột 2 lần vào ảnh để chọn điểm trên và dưới vật thể (để đo pixel height).")
    print("❌ Nhấn ESC để thoát.\n")

    while True:
        display = img.copy()
        for pt in click_points:
            cv2.circle(display, pt, 5, (0, 0, 255), -1)

        cv2.imshow("Chọn 2 điểm: đỉnh & chân vật thể", display)
        key = cv2.waitKey(1)
        if key == 27:  # ESC để thoát
            break

    cv2.destroyAllWindows()

    # Hiển thị kết quả
    if samples:
        print("\n📊 Kết quả các mẫu đã lưu:")
        for i, (px, rh, td, alpha) in enumerate(samples, 1):
            print(f"Mẫu {i}: pixel={px}, height={rh}m, distance={td}m, alpha={alpha:.4f}")

        # Tính alpha trung bình
        alpha_mean = np.mean([s[3] for s in samples])
        print(f"\n🎯 Hệ số hiệu chỉnh alpha trung bình: {round(alpha_mean, 4)}")

        # Lưu ra file CSV nếu muốn
        save = input("💾 Bạn có muốn lưu mẫu ra file CSV? (y/n): ").lower()
        if save == 'y':
            import csv
            with open("distance_samples.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["pixel_height", "real_height", "true_distance", "alpha"])
                writer.writerows(samples)
            print("✅ Đã lưu file distance_samples.csv")

if __name__ == "__main__":
    main()
