# MULTI-DOMAIN AI RECOMMENDATION ARCHITECTURE SPECIFICATION

**Hệ thống:** CineAI V2.0 + MusicAI Unified Platform  
**Phiên bản:** 1.0  
**Giai đoạn:** Phase M6 — Production Architecture  
**Tác giả:** Senior Full-Stack Software Architect, ML Systems Engineer  

---

## 1. TỔNG QUAN KIẾN TRÚC TOÀN HỆ THỐNG

Hệ thống được thiết kế theo mô hình **Đa miền Thống nhất (Unified Multi-Domain Micro-Monolith)**. Hai miền ứng dụng độc lập:
1. **CineAI:** Gợi ý phim theo xếp hạng minh thị (Explicit feedback: ratings 1–5 sao trên MovieLens 100K) gồm **10 mô hình AI**.
2. **MusicAI:** Gợi ý bài hát theo tương tác ngầm định (Implicit feedback: play counts trên Million Song Dataset Taste Profile 10-core) gồm **6 mô hình AI**.

Cả hai miền chạy đồng thời trên cùng một hạ tầng FastAPI backend duy nhất và chia sẻ chung một ứng dụng React Frontend với cơ chế chuyển đổi ngữ cảnh tức thì (Domain Context Switching).

```
+---------------------------------------------------------------------------------------+
|                                    CLIENT BROWSER                                     |
|    Desktop (1440px)  |  Tablet (768px)  |  Mobile (375px) | SPA Responsive PWA        |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            | (HTTP/JSON, Port 3000 / 8080)
                                            v
+---------------------------------------------------------------------------------------+
|                                  NGINX REVERSE PROXY                                  |
|   - SPA Static Hosting (/assets/, /index.html)                                        |
|   - Security Headers (X-Frame-Options, X-Content-Type, CSP, Strict Referrer)          |
|   - Gzip Compression for JSON, JS, CSS, SVG                                           |
|   - Reverse Proxy: /api/* -> http://backend:8000/api/*                                |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            | (Internal Docker Network / Proxy)
                                            v
+---------------------------------------------------------------------------------------+
|                         FASTAPI UNIFIED APPLICATION SERVER                             |
|                                (Uvicorn ASGI, Port 8000)                              |
|                                                                                       |
|   +------------------------------------+  +---------------------------------------+   |
|   |         CineAI Service Router      |  |         MusicAI Service Router        |   |
|   |        (/api/v1/recommendations)   |  |        (/api/music/recommendations)   |   |
|   |        (/api/v1/movies, /models)   |  |        (/api/music/songs, /history)   |   |
|   +-----------------+------------------+  +-------------------+-------------------+   |
|                     |                                         |                       |
|   +-----------------v------------------+  +-------------------v-------------------+   |
|   |       CineAI Model Registry        |  |        MusicAI Model Registry         |   |
|   |  - Lazy Checkpoint Loading         |  |  - Lazy Checkpoint Loading            |   |
|   |  - Rating Matrix Indices           |  |  - Song-Level Inverted Indices        |   |
|   |  - In-memory Metadata Cache        |  |  - Fast Candidate Masking (Dedupe)    |   |
|   +-----------------+------------------+  +-------------------+-------------------+   |
|                     |                                         |                       |
|   +-----------------v------------------+  +-------------------v-------------------+   |
|   |        10 CineAI AI Models         |  |          6 MusicAI AI Models          |   |
|   | 1. Popularity Baseline             |  | 1. Popularity Frequency Baseline      |   |
|   | 2. User-Based CF (Pearson)         |  | 2. Item-Based CF (Adjusted Cosine)    |   |
|   | 3. Item-Based CF (Cosine)          |  | 3. User-Based CF (Taste Vectors)      |   |
|   | 4. KNN Baseline with Bias          |  | 4. Implicit ALS (Hu-Koren-Volinsky)   |   |
|   | 5. SVD (Funk Matrix Factorization) |  | 5. BPR-MF (Pairwise Margin Loss)      |   |
|   | 6. SVD++ (Implicit History Bias)   |  | 6. Implicit NeuMF (PyTorch MLP+GMF)   |   |
|   | 7. NMF (Non-negative MF)           |  |                                       |   |
|   | 8. TF-IDF Content-Based            |  |                                       |   |
|   | 9. Neural CF (He et al.)           |  |                                       |   |
|   | 10. Hybrid Dynamic Weighted        |  |                                       |   |
|   +-----------------+------------------+  +-------------------+-------------------+   |
+---------------------+-----------------------------------------+-----------------------+
                      |                                         |
                      | (Read-Only Mount)                       | (Read-Only Mount)
                      v                                         v
+----------------------------------------+  +-------------------------------------------+
|          CINEAI STORAGE VOLUMES        |  |           MUSICAI STORAGE VOLUMES         |
|  - data/processed/movies_clean.csv     |  |  - data/music/processed/songs.csv         |
|  - data/processed/train.csv            |  |  - data/music/processed/train.csv         |
|  - models/checkpoints/*.pkl (10 files) |  |  - models/checkpoints/music/*.pkl (6 files|
+----------------------------------------+  +-------------------------------------------+
```

---

## 2. NGUYÊN TẮC CÁCH LY VÀ ĐỒNG NHẤT (ISOLATION & COEXISTENCE)

1. **Không xung đột Không gian Tên (Namespace Isolation):**
   - CineAI sử dụng prefix `/api/v1` (`/api/v1/recommendations`, `/api/v1/movies`, `/api/v1/models`).
   - MusicAI sử dụng prefix `/api/music` và `/api/v1/music` (`/api/music/recommendations`, `/api/music/songs`, `/api/music/users/{id}/history`).
   - Cả hai router hoàn toàn tách biệt, không phụ thuộc vào trạng thái toàn cục của nhau.

2. **Quản lý Mô hình Thông minh (Lazy Loading & Single-Instance Registry):**
   - Checkpoint mô hình không được nạp đồng loạt khi khởi động container để tiết kiệm tối đa RAM và thời gian khởi động (Startup latency < 2 giây).
   - Mỗi mô hình chỉ được nạp một lần duy nhất vào bộ nhớ khi có yêu cầu suy luận đầu tiên (Lazy Loading on demand). Các yêu cầu tiếp theo tái sử dụng thể hiện mô hình đã nạp trong bộ nhớ đệm (Warm inference latency < 15ms).
   - Tuyệt đối không huấn luyện lại mô hình trong vòng đời xử lý HTTP request.

3. **Cơ chế Graceful Fallback & Cold-Start Resilience:**
   - Đối với người dùng mới (chưa có trong tập huấn luyện), hệ thống không báo lỗi 500 mà tự động kích hoạt cờ fallback `fallback_used: true`, chuyển sang mô hình `popularity` baseline và thông báo nguyên nhân minh bạch cho người dùng.

4. **Kiểm soát Tương tranh Đa luồng (OpenBLAS Thread Throttling):**
   - Biến môi trường `OPENBLAS_NUM_THREADS=1` được áp dụng bắt buộc trên toàn hệ thống để tránh tình trạng OpenBLAS tranh chấp tài nguyên bộ nhớ trên môi trường Windows hoặc Container đa luồng.
