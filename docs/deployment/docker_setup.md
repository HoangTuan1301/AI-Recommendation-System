# HƯỚNG DẪN THIẾT LẬP VÀ VẬN HÀNH DOCKER (DOCKER SETUP GUIDE)

**Hệ thống:** CineAI V2.0 + MusicAI Unified Platform  
**Phiên bản:** 1.0  
**Giai đoạn:** Phase M6  

---

## 1. YÊU CẦU TIỀN ĐIỀU KIỆN

1. Đã cài đặt **Docker Engine** (phiên bản 24.0+) hoặc **Docker Desktop**.
2. Đã cài đặt **Docker Compose** v2 (hoặc `docker-compose`).
3. Đã chuẩn bị đầy đủ 16 tệp checkpoint AI và các tệp dữ liệu đã xử lý (`data/processed` và `data/music/processed`).

> [!IMPORTANT]
> **Chiến lược Cung cấp Dữ liệu và Checkpoints:**  
> Toàn bộ các checkpoint AI (`models/checkpoints/*.pkl`) và dữ liệu đã xử lý (`data/processed/*.csv`, `data/music/processed/*.csv`) **KHÔNG** được đóng gói trực tiếp vào Docker image để tránh làm phình to image (tiết kiệm hàng trăm MB dung lượng mạng).  
> Thay vào đó, chúng được gắn kết an toàn vào container dưới dạng **Read-Only Volume Mounts** thông qua `docker-compose.yml`.

---

## 2. KIỂM TRA TÍNH TOÀN VẸN CỦA ARTIFACTS TRƯỚC KHI BUILD DOCKER

Trước khi khởi động Docker, chạy lệnh kiểm tra tính toàn vẹn và SHA-256 của tất cả các checkpoint và dataset:

```bash
# Trên Windows PowerShell
.\.venv\Scripts\python scripts/verify_production_artifacts.py

# Trên Linux / macOS
python scripts/verify_production_artifacts.py
```

Xác nhận kết quả hiển thị: `OVERALL ARTIFACT STATUS: ALL VERIFIED (PASS)`.

---

## 3. CÁC TỆP DOCKER TRONG DỰ ÁN

| Tệp tin | Vai trò | Đặc điểm kỹ thuật |
|:---|:---|:---|
| [`Dockerfile.backend`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/Dockerfile.backend) | Xây dựng image Backend FastAPI | Python 3.11-slim, User không đặc quyền `appuser` (UID 10001), tối ưu hóa `OPENBLAS_NUM_THREADS=1`, health check tích hợp. |
| [`Dockerfile.frontend`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/Dockerfile.frontend) | Xây dựng image Frontend React | Multi-stage build (Node 20 Alpine builder -> Nginx unprivileged Alpine runner), code-splitting chunking, gzip compression, SPA fallback routing. |
| [`frontend/nginx.conf`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/frontend/nginx.conf) | Cấu hình máy chủ Nginx | Cổng 8080, Reverse proxy `/api/*` tới backend, bổ sung 4 header bảo mật chuẩn OWASP. |
| [`docker-compose.yml`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/docker-compose.yml) | Điều phối 2 dịch vụ | Mạng nội bộ `recommendation-net`, gắn kết volume read-only, tự động khởi động lại, kiểm soát thứ tự khởi động với health checks. |
| [`.dockerignore`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/.dockerignore) | Loại bỏ tệp dư thừa | Loại bỏ `.git`, `.venv`, `node_modules`, cache, và dữ liệu thô dung lượng lớn. |

---

## 4. QUY TRÌNH XÂY DỰNG VÀ KHỞI ĐỘNG HỆ THỐNG

### Bước 1: Build Docker Images
```bash
docker compose build
```

### Bước 2: Khởi động Containers ở chế độ nền (Detached)
```bash
docker compose up -d
```

### Bước 3: Kiểm tra trạng thái và Health Check
```bash
docker compose ps
```
Cả hai container phải hiển thị trạng thái `Up (healthy)`:
- `multidomain-ai-backend`: Port 8000
- `multidomain-ai-frontend`: Port 3000

### Bước 4: Xem Logs thời gian thực
```bash
# Xem logs toàn hệ thống
docker compose logs -f

# Xem riêng logs backend
docker compose logs -f backend

# Xem riêng logs frontend
docker compose logs -f frontend
```

### Bước 5: Truy cập Ứng dụng
- **Giao diện Người dùng (React Multi-Domain):** `http://localhost:3000`
- **FastAPI OpenAPI Interactive Docs:** `http://localhost:8000/docs`
- **API Health Endpoint:** `http://localhost:8000/api/v1/health`

---

## 5. DỪNG VÀ DỌN DẸP CONTAINER

```bash
# Dừng các container (dữ liệu trên host được bảo toàn nguyên vẹn)
docker compose down

# Dừng và xóa cả volumes/mạng nếu cần dọn dẹp triệt để
docker compose down --volumes --remove-orphans
```
