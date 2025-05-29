

## 📸 Camera Calibration - Hướng dẫn hiệu chỉnh camera

Hiệu chỉnh camera giúp xác định ma trận nội tại (camera matrix) và hệ số méo (distortion coefficients), từ đó tính toán chính xác khoảng cách từ hình ảnh.

---

### ✅ **Bước 1: Chụp ảnh Checkerboard (bàn cờ)**

> File sử dụng: `calibpre.py`

#### ✔️ Cách dùng:

1. In một tấm **bàn cờ 7x6 điểm góc** (tương đương 8x7 ô vuông).
2. Đặt bàn cờ ở nhiều góc và khoảng cách khác nhau trước webcam.
3. Chạy script:

```bash
python calibpre.py
```

4. Giao diện hiện ra, nếu thấy `"Checkerboard FOUND"`:

   * Nhấn **`s`** để lưu ảnh.
   * Nhấn **`q`** để thoát sau khi đã có khoảng 15–20 ảnh.

🗂 Ảnh được lưu vào thư mục `calib_images/`.

---

### ✅ **Bước 2: Tính toán ma trận camera và hệ số méo**

> File sử dụng: `calib.py`

#### ✔️ Cách dùng:

1. Đảm bảo bạn đã có ảnh trong thư mục `calib_images/`.
2. Kiểm tra lại thông số trong code:

   * `CHECKERBOARD = (7, 6)` → số điểm góc
   * `square_size = 25.0` → kích thước thật mỗi ô vuông (đơn vị: **mm** hoặc **cm**, miễn nhất quán)
3. Chạy:

```bash
python calib.py
```

📦 Kết quả:

* In ra: `camera_matrix` và `dist_coeffs`
* Tự động lưu file `camera_calib.npz` để dùng cho các ứng dụng khác

---

### ✅ **Bước 3: Ước lượng hệ số hiệu chỉnh khoảng cách (alpha)**

> File sử dụng: `alpha_caculated.py`

#### ✔️ Mục đích:

Xác định hệ số `alpha` giúp hiệu chỉnh sai số khi tính khoảng cách từ bounding box và tiêu cự.

#### ✔️ Cách dùng:

1. Chuẩn bị một ảnh có vật thể đứng thẳng (ví dụ: người, xe), có kích thước thật và khoảng cách đo được.
2. Mở file:

```bash
python alpha_caculated.py
```

3. Giao diện ảnh sẽ mở:

   * **Click chuột 2 lần** để chọn đỉnh và chân của vật thể.
   * Nhập:

     * Chiều cao thật (m)
     * Khoảng cách thật (m)

4. Lặp lại nhiều lần để lấy nhiều mẫu

📊 Cuối cùng:

* In ra danh sách mẫu và `alpha trung bình`
* Có thể lưu thành CSV nếu muốn

---

## ✅ Kết quả đầu ra:

* `camera_calib.npz`: Chứa `camera_matrix` và `dist_coeffs`
* `distance_samples.csv`: (tuỳ chọn) chứa các mẫu `pixel_height`, `real_height`, `true_distance`, `alpha`
* `alpha_mean`: hệ số hiệu chỉnh bạn sẽ dùng trong các mô hình tính khoảng cách từ bounding box.

---

## 🛠 Sử dụng trong inference (dự đoán)

Khi bạn đã có:

* `camera_matrix`
* `dist_coeffs`
* `alpha`

Thì có thể dùng để:

* **undistort ảnh** khi hiển thị
* **tính khoảng cách** từ bounding box theo công thức:

```python
distance = ALPHA * (real_height * focal_length) / pixel_height
```

