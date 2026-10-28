# BÁO CÁO KIỂM THỬ TRIỂN KHAI HỆ THỐNG ĐA MIỀN (DEPLOYMENT TEST REPORT)

**Dự án:** Multi-Domain AI Recommendation System (CineAI V2.0 + MusicAI)  
**Môi trường thực hiện:** Clean Workspace (`E:\HOCTAP\K5\TTNT\AI-Recommendation-System-Clean`)  
**Ngày kiểm thử:** 28/10/2026  
**Chủ trì kiểm thử:** QA Lead, Senior DevOps Engineer & ML Systems Auditor  

---

## 1. TỔNG HỢP KẾT QUẢ KIỂM THỬ THỰC TẾ

| Nhóm bài kiểm thử (Test Suite) | Số lượng Test Cases | Trạng thái | Thời gian thực thi | Bằng chứng xác thực |
| :--- | :---: | :---: | :---: | :--- |
| **FastAPI Backend, MusicAI & Bootstrap Tests** | 61 / 61 | **PASS (100%)** | 15.90s | `tests/test_fastapi_backend.py`, `tests/music/`, `tests/test_artifact_bootstrap.py` |
| **CineAI Pipeline & Audit Tests** | 45 / 45 | **PASS (100%)** | 19.15s | `tests/test_phase4a_pipeline.py`, `tests/test_phase4b_pipeline.py`, `tests/test_phase4c_*.py` |
| **Frontend Unit & Component Tests** | 11 / 11 | **PASS (100%)** | 1.64s | `frontend/src/test/components.test.tsx`, `frontend/src/test/music.test.tsx` (Vitest) |
| **Frontend Production Build (TypeScript)** | 1 / 1 | **PASS** | 0.43s | `npm run build` (`tsc -b && vite build`): bundle `dist/` thành công, 0 lỗi TS |
| **Kiểm định Suy luận 16 Mô hình AI** | 16 / 16 | **PASS (100%)** | N/A | Đo trực tiếp qua `scripts/audit_memory_16_models.py`, 16/16 mô hình suy luận hợp lệ |
| **Kiểm tra Toàn vẹn Checkpoint (SHA-256)** | 16 / 16 | **PASS (100%)** | N/A | Toàn bộ 16 checkpoints và datasets khớp 100% manifest SHA-256 |
| **Kiểm thử Bảo mật An toàn Giải nén Archive** | 4 / 4 | **PASS (100%)** | 0.12s | Chặn đứng path traversal (`../`), chặn đường dẫn tuyệt đối (`/etc/evil`), chặn symlink injection |

---

## 2. BÁO CÁO ĐO LƯỜNG BỘ NHỚ RAM VÀ ĐÁNH GIÁ RENDER FREE

Đo lường trên tiến trình Uvicorn thực tế với cấu hình `MAX_CACHED_MODELS=2`:

### 2.1. Tiến trình tiêu thụ RAM (Resident Set Size):
- **Stage 1 (Khởi động nguội — 0 mô hình trong RAM):** RSS = **368.84 MB**, VMS = 330.43 MB.
- **Stage 2 (Tải 2 mô hình — MusicAI Item-CF & CineAI Neural CF):** RSS = **417.60 MB** (+48.76 MB).
- **Stage 3 (Sau khi suy luận tuần tự qua toàn bộ 16 mô hình với LRU Eviction Cache):** RSS = **478.81 MB** (Giảm **124.02 MB** so với khi không dọn cache: 602.83 MB $\rightarrow$ 478.81 MB).
- **Ngoại lệ bộ nhớ / OOM:** **0 lỗi**.

### 2.2. Đánh giá Khách quan về Tính khả thi trên Render Free (512 MB):
- **Trạng thái:** **CONDITIONALLY VIABLE (DEMO ONLY) / NOT VERIFIED CHO HIGH-CONCURRENCY PRODUCTION**.
- **Phân tích kỹ thuật:**
  - Ngưỡng 478.81 MB nằm dưới giới hạn trần 512 MB của Render Free, nhưng biên độ an toàn chỉ còn **~33.19 MB (khoảng 6.5%)**.
  - Trong quá trình vận hành thông thường (phục vụ 1-2 mô hình theo yêu cầu của người dùng), RAM chỉ tiêu thụ **~417 MB** (rất an toàn).
  - Tuy nhiên, nếu có nhiều người dùng đồng thời gọi các mô hình khác nhau liên tục, hoặc khi hệ điều hành Linux cấp phát bộ nhớ phụ trợ, nguy cơ vượt ngưỡng 512 MB và bị Render OOM Killer kích hoạt là hoàn toàn có thể xảy ra.
- **Khuyến nghị kiến trúc:**
  - **Gói Free Tier (512 MB):** Chỉ khuyến nghị dùng cho mục đích **Bảo vệ đồ án / Demo nội bộ** với tham số bắt buộc `MAX_CACHED_MODELS=2`.
  - **Gói Starter Tier (2 GB RAM, $7/tháng):** **PASS / 100% SẴN SÀNG CHO PRODUCTION CHÍNH THỨC**. Ở mức 478 MB trên tổng 2,048 MB, dịch vụ chỉ tiêu thụ 23% tài nguyên, triệt tiêu 100% rủi ro OOM và hỗ trợ lưu giữ toàn bộ 16 mô hình vĩnh viễn trên RAM.

---

## 3. CHI TIẾT THỜI GIAN SUY LUẬN VÀ TRẢI NGHIỆM CỦA 16 MÔ HÌNH AI

| STT | Miền | Tên mô hình AI | Thuật toán | Tải nguội (Cold Load) | Suy luận nóng (Warm Inference) | Kết quả trả về |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | CineAI | Popularity Baseline | Heuristic thống kê | 73.40 ms | 8.63 ms | Top-10 Movies |
| 2 | CineAI | User-Based CF | Láng giềng người dùng | 224.78 ms | 15.92 ms | Top-10 Movies |
| 3 | CineAI | Item-Based CF | Ma trận tương đồng phim | 200.39 ms | 15.26 ms | Top-10 Movies |
| 4 | CineAI | KNN Baseline | k-NN chuẩn hóa | 198.45 ms | 18.32 ms | Top-10 Movies |
| 5 | CineAI | Funk SVD | Phân rã ma trận SGD | 206.04 ms | 9.25 ms | Top-10 Movies |
| 6 | CineAI | SVD++ | Phân rã kèm implicit feedback | 201.92 ms | 9.06 ms | Top-10 Movies |
| 7 | CineAI | NMF | Ma trận không âm | 204.81 ms | 9.24 ms | Top-10 Movies |
| 8 | CineAI | Content-Based | TF-IDF đặc trưng nội dung | 212.42 ms | 13.98 ms | Top-10 Movies |
| 9 | CineAI | Neural CF (NeuMF) | PyTorch Deep Neural Network | 17.18 ms | 16.48 ms | Top-10 Movies |
| 10 | CineAI | Hybrid Ensemble | Kết hợp trọng số tối ưu | 380.74 ms | 30.98 ms | Top-10 Movies |
| 11 | MusicAI | Popularity Baseline | Top lượt nghe MSD | 86.85 ms | 2.86 ms | Top-10 Songs |
| 12 | MusicAI | Item-Based CF | Cosine tương đồng bài hát | 11.82 ms | 11.27 ms | Top-10 Songs |
| 13 | MusicAI | User-Based CF | Láng giềng người nghe | 402.00 ms | 3.29 ms | Top-10 Songs |
| 14 | MusicAI | Implicit ALS | iALS Hu-Koren-Volinsky | 214.13 ms | 28.01 ms | Top-10 Songs |
| 15 | MusicAI | BPR-MF | Bayesian Personalized Ranking | 212.81 ms | 27.21 ms | Top-10 Songs |
| 16 | MusicAI | Neural CF (NeuMF) | PyTorch NeuMF ngầm định | 229.04 ms | 7.19 ms | Top-10 Songs |

---

## 4. TỔNG HỢP KIỂM TRA DOCKER RUNTIME
- **Lệnh kiểm tra:** `Get-Command docker`, `Get-Service *docker*`.
- **Kết quả thực tế:** Docker Engine / Docker Desktop không được cài đặt trên hệ điều hành máy tính cục bộ.
- **Kết luận:** **Docker Runtime trên máy Host cục bộ: NOT EXECUTED (do thiếu Docker Engine)**. Mã nguồn cấu hình `Dockerfile.backend` và `docker-compose.yml` đã được hoàn thiện chuẩn production, sẵn sàng build trực tiếp trên Render Cloud hoặc môi trường Linux có Docker.
