# HƯỚNG DẪN TRIỂN KHAI BACKEND LÊN RENDER (RENDER DEPLOYMENT GUIDE)

**Dự án:** Multi-Domain AI Recommendation System (CineAI V2.0 + MusicAI)  
**Tác giả:** Đội ngũ Kỹ sư AI & DevOps K5  
**Mục tiêu:** Hướng dẫn từng bước triển khai FastAPI Backend (chạy 16 mô hình AI) lên nền tảng Render bằng Docker Container.

---

## 1. PHÂN TÍCH TÍNH KHẢ THI VÀ GIỚI HẠN TÀI NGUYÊN TRÊN RENDER

### 1.1. So sánh giữa Gói Miễn phí (Free Tier) và Gói Trả phí (Starter Tier)

| Tiêu chí | Gói Miễn phí (Free Tier) | Gói Trả phí (Starter Tier) - Khuyến nghị |
| :--- | :---: | :---: |
| **Chi phí** | $0 / tháng | $7 / tháng |
| **RAM cung cấp** | **512 MB** | **2,048 MB (2 GB)** |
| **CPU cung cấp** | 0.1 vCPU (chia sẻ) | 1.0 vCPU |
| **Cơ chế Sleep (Ngủ)** | Tự động ngủ sau **15 phút** không có request | **Chạy 24/7**, không bao giờ ngủ |
| **Thời gian Khởi động nguội (Cold Start)** | **50 – 75 giây** khi có request đánh thức | **0 giây** (luôn trực tuyến) |
| **Khả năng chứa 16 mô hình trong RAM** | Đòi hỏi cơ chế **LRU Eviction** (`MAX_CACHED_MODELS=3-4`) để giữ RAM ~417–450 MB. | Thoải mái chứa toàn bộ 16 mô hình đồng thời (tiêu thụ ~600 MB / 2,048 MB = 29% RAM). |
| **Đánh giá triển khai** | **Phù hợp cho Demo đồ án / Chấm điểm** nếu chấp nhận cold-start 50s ban đầu. | **Tối ưu tuyệt đối cho Production** nếu nhóm có ngân sách $7. |

### 1.2. Chiến lược Quản lý Bộ nhớ trên Gói Free Tier (512 MB RAM)
- Hệ thống áp dụng **Lazy Loading** (khi khởi động chỉ tốn ~368 MB).
- Khi có request, chỉ nạp mô hình được yêu cầu (Item-CF +50 MB $\rightarrow$ ~417 MB).
- Đặt biến môi trường `MAX_CACHED_MODELS=4` để hệ thống tự động giải phóng (evict) mô hình cũ và thu hồi bộ nhớ qua `gc.collect()`, ngăn chặn tuyệt đối lỗi **OOM (Out Of Memory)**.

---

## 2. CƠ CHẾ PHÂN PHỐI VÀ NẠP MODEL ARTIFACTS TRÊN RENDER

Do Render build Docker container trực tiếp từ mã nguồn GitHub (không mount volume máy cá nhân), các tệp checkpoint và dữ liệu cần cơ chế cung cấp tự động:

1. **Các tệp đã có sẵn trên GitHub (CineAI Checkpoints):**
   - 10 checkpoint phim tại `models/checkpoints/*.pkl` (~70 MB) đã được commit trên nhánh `main`, Dockerfile tự động sao chép vào image.
2. **Các tệp dữ liệu đã xử lý và checkpoint MusicAI (chưa commit trên Git):**
   - Đã được đóng gói thành tệp nén tinh gọn: `artifacts/production_artifacts.tar.gz` (**dung lượng chỉ 8.49 MB**).
   - Mã băm SHA-256 xác thực: `649974a0b43645105ad5718271524f434cd242701bf04674e7e5aa9849a7cbd8`.
3. **Cơ chế nạp tự động (Zero-Manual-Intervention):**
   - Khi container khởi động, script [`scripts/ensure_artifacts.py`](file:///E:/HOCTAP/K5/TTNT/AI-Recommendation-System-Clean/scripts/ensure_artifacts.py) kiểm tra sự tồn tại của các tệp.
   - Nếu phát hiện tệp còn thiếu, script sẽ tự động tải `production_artifacts.tar.gz` từ biến môi trường `ARTIFACTS_DOWNLOAD_URL`, giải nén và kiểm tra SHA-256 trước khi Uvicorn lắng nghe cổng.

---

## 3. CÁC BƯỚC TRIỂN KHAI TỪNG BƯỚC TRÊN RENDER (STEP-BY-STEP)

### Bước 1: Tạo Web Service mới trên Render
1. Đăng nhập vào [Render Dashboard](https://dashboard.render.com/).
2. Nhấn nút **New +** ở góc trên bên phải $\rightarrow$ Chọn **Web Service**.
3. Chọn **Build and deploy from a Git repository** $\rightarrow$ Kết nối tài khoản GitHub và chọn repository:  
   `https://github.com/HoangTuan1301/AI-Recommendation-System`

### Bước 2: Cấu hình thông số triển khai (Settings)
Điền chính xác các giá trị cấu hình sau:

| Mục cấu hình | Giá trị thiết lập | Ghi chú |
| :--- | :--- | :--- |
| **Name** | `cineai-musicai-backend` | Tên dịch vụ trên Render |
| **Region** | `Singapore (Southeast Asia)` | Vùng địa lý gần Việt Nam nhất để giảm độ trễ |
| **Branch** | `main` | Nhánh chứa mã nguồn sạch |
| **Language / Runtime** | **Docker** | Render sẽ tự phát hiện Dockerfile |
| **Docker Build Context Directory** | `.` | Thư mục gốc của repository |
| **Dockerfile Path** | `./Dockerfile.backend` | Đường dẫn tới Dockerfile production của backend |
| **Instance Type** | `Free` (hoặc `Starter` nếu có tài trợ) | 512 MB RAM cho Free |

### Bước 3: Cấu hình Biến môi trường (Environment Variables)
Tại mục **Environment Variables**, bấm **Add Environment Variable** và thêm các khóa sau:

| Tên biến (Key) | Giá trị (Value) | Giải thích |
| :--- | :--- | :--- |
| `OPENBLAS_NUM_THREADS` | `1` | Khống chế đơn luồng CPU, chống tràn bộ nhớ |
| `MKL_NUM_THREADS` | `1` | Tối ưu tài nguyên toán học ma trận |
| `OMP_NUM_THREADS` | `1` | Hạn chế đa luồng OpenMP |
| `PYTHONUNBUFFERED` | `1` | Xuất log console tức thì ra Render Dashboard |
| `MAX_CACHED_MODELS` | `2` | Giới hạn tối đa 2 mô hình mỗi domain lưu trên RAM (bắt buộc cho Free Tier) |
| `CORS_ORIGINS` | `https://<ten-ung-dung-cua-ban>.vercel.app,http://localhost:5173` | Danh sách domain được phép gọi API |
| `ARTIFACTS_DOWNLOAD_URL` | *URL tải tệp `production_artifacts.tar.gz`* | Link GitHub Release Asset hoặc Cloud Storage trực tiếp |

### Bước 4: Cấu hình Health Check
Tại mục **Advanced**:
- **Health Check Path:** `/api/v1/health`
- Render sẽ định kỳ gọi endpoint này để xác nhận container hoạt động bình thường trước khi định tuyến người dùng.

### Bước 5: Bấm "Create Web Service"
- Render sẽ bắt đầu quá trình `docker build`, cài đặt dependencies, sao chép code và chạy `scripts/ensure_artifacts.py`.
- Khi log hiển thị:
  ```text
  [Artifacts] All required production datasets and checkpoints verified present. Ready.
  INFO:     Uvicorn running on http://0.0.0.0:10000 (Press CTRL+C to quit)
  ```
- Dịch vụ sẽ chuyển sang trạng thái **Live**. URL của bạn sẽ có dạng:  
  `https://cineai-musicai-backend.onrender.com`

---

## 4. KIỂM THỬ XÁC NHẬN SAU KHI DEPLOY
Sau khi Render báo **Live**, thực hiện các lệnh kiểm tra:

1. **Kiểm tra Health tổng thể:**
   ```bash
   curl -s https://cineai-musicai-backend.onrender.com/api/v1/health
   ```
   *Kết quả mong đợi:* `{"status":"healthy","catalog_movies":1682,"available_models":10}`
2. **Kiểm tra Health MusicAI:**
   ```bash
   curl -s https://cineai-musicai-backend.onrender.com/api/music/health
   ```
   *Kết quả mong đợi:* `{"status":"healthy","domain":"music","total_songs":2706,...}`
3. **Thử nghiệm gợi ý thực tế:**
   ```bash
   curl -X POST https://cineai-musicai-backend.onrender.com/api/v1/recommendations \
     -H "Content-Type: application/json" \
     -d '{"user_id": 196, "model_id": "neural_cf", "top_n": 5}'
   ```
