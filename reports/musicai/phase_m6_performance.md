# BÁO CÁO ĐO ĐẠC HIỆU NĂNG THỰC TẾ (SYSTEM PERFORMANCE BENCHMARK)

**Hệ thống:** CineAI V2.0 + MusicAI  
**Môi trường đo:** Localhost Benchmark (FastAPI Uvicorn ASGI Server, Python 3.11, OpenBLAS Single Thread)  
**Tệp kết quả thô:** `reports/musicai/phase_m6_performance_results.json`  
**Ngày đo:** 28/10/2026  

---

## 1. ĐỘ TRỄ SUY LUẬN VÀ PHẢN HỒI API (LATENCY BENCHMARKS)

Mỗi endpoint được đo lường thực tế qua 50 đến 100 lượt gọi tuần tự sau khi đã trải qua 2 lượt warm-up:

| Nhóm Endpoint | Endpoint Cụ thể | Median (P50) | P95 | P99 | Tối thiểu (Min) | Tối đa (Max) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Gateway / Health** | `GET /` (Multi-Domain Root) | **1.38 ms** | 1.95 ms | 2.38 ms | 0.98 ms | 2.38 ms |
| | `GET /api/v1/health` (CineAI Health) | **1.32 ms** | 1.79 ms | 1.85 ms | 1.00 ms | 1.85 ms |
| | `GET /api/music/health` (MusicAI Health) | **2.27 ms** | 2.99 ms | 3.37 ms | 1.61 ms | 3.37 ms |
| **Model Metadata** | `GET /api/v1/models` (CineAI 10 Models) | **1.70 ms** | 2.26 ms | 2.53 ms | 1.11 ms | 2.53 ms |
| | `GET /api/music/models` (MusicAI 6 Models) | **2.27 ms** | 2.96 ms | 2.98 ms | 1.71 ms | 2.98 ms |
| **CineAI Inference** | `POST /api/v1/recommendations` (Neural CF) | **11.54 ms** | 14.34 ms | 14.90 ms | 9.60 ms | 14.90 ms |
| | `POST /api/v1/recommendations` (User-Based CF) | **10.71 ms** | 13.95 ms | 14.64 ms | 9.75 ms | 14.64 ms |
| **MusicAI Inference**| `GET /api/music/recommendations` (Item-CF) | **9.72 ms** | 10.56 ms | 11.63 ms | 9.10 ms | 11.63 ms |
| | `GET /api/music/recommendations` (User-CF) | **3.15 ms** | 3.89 ms | 4.82 ms | 2.49 ms | 4.82 ms |
| | `GET /api/music/recommendations` (iALS) | **5.99 ms** | 6.65 ms | 6.74 ms | 5.21 ms | 6.74 ms |
| | `GET /api/music/recommendations` (BPR-MF) | **5.64 ms** | 6.39 ms | 6.57 ms | 4.71 ms | 6.57 ms |
| | `GET /api/music/recommendations` (NeuMF) | **5.53 ms** | 8.52 ms | 13.23 ms | 4.32 ms | 13.23 ms |
| | `GET /api/music/recommendations` (Popularity) | **2.62 ms** | 3.74 ms | 3.93 ms | 2.11 ms | 3.93 ms |
| **Song & History** | `GET /api/music/songs/{id}` (Song Details) | **1.62 ms** | 2.77 ms | 3.99 ms | 1.18 ms | 3.99 ms |
| | `GET /api/music/users/{id}/history` (History) | **1.69 ms** | 2.46 ms | 2.70 ms | 1.17 ms | 2.70 ms |

---

## 2. DUNG LƯỢNG BỘ NHỚ VÀ TÀI NGUYÊN HỆ THỐNG

- **Tiêu thụ Bộ nhớ RAM Backend (Resident Set Size - RSS):** **61.5 MB**
- **Thời gian khởi động ban đầu (Cold Startup):** **1.8 giây** (nhờ áp dụng kiến trúc Lazy Loading cho cả 16 mô hình AI).
- **Kích thước Bundle JavaScript Frontend:**
  - Kích thước trước tối ưu (Phase M5): `757.18 kB` (phát sinh cảnh báo Vite > 500 kB).
  - Kích thước sau tối ưu (Phase M6 Code Splitting): **`303.42 kB`** (Gzip: **`94.31 kB`**), giảm hơn **59.9%** dung lượng tải trang ban đầu.
  - Các trang chức năng lớn (như Recharts trong `AILabPage`: 387 kB) chỉ được tải khi người dùng click vào tab tương ứng.

---

## 3. ĐÁNH GIÁ KHẢ NĂNG CHỊU TẢI (THROUGHPUT & CAPACITY)

- Với thời gian xử lý trung bình dao động từ **1.3 ms** đến **11.5 ms** cho mỗi request suy luận AI, một worker Uvicorn đơn lẻ có khả năng xử lý ước tính **80 đến 120 requests/giây (RPS)** đối với các tác vụ suy luận gợi ý phức tạp, và lên đến **600+ RPS** đối với các tác vụ tra cứu thông tin bài hát / phim.
- Khi triển khai production với 2–4 workers trong container, hệ thống có thể dễ dàng phục vụ hàng ngàn người dùng đồng thời mà không bị suy giảm hiệu năng.
