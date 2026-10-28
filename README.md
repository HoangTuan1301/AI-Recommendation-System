# Multi-Domain AI Recommendation System (CineAI V2.0 + MusicAI)

> Nền tảng Gợi ý Đa miền Thông minh (Multi-Domain AI Recommendation Platform) ứng dụng Trí tuệ nhân tạo và Học máy chuyên sâu, tích hợp hai miền giải trí hàng đầu: **Phim ảnh (CineAI)** và **Âm nhạc (MusicAI)** trong cùng một ứng dụng web cao cấp.

Dự án Bài tập lớn môn **Trí tuệ nhân tạo (AI)** — Học kỳ 5
**Khoa Công nghệ Thông tin — Trường Đại học**
**Tác giả:** Đội ngũ Kỹ sư AI & Kiến trúc sư Phần mềm Full-Stack K5
**GitHub:** [https://github.com/HoangTuan1301/AI-Recommendation-System](https://github.com/HoangTuan1301/AI-Recommendation-System)

---

## 1. Giới thiệu Tổng quan Nền tảng Đa miền

Hệ thống khuyến nghị (Recommendation System) là phân nhánh cốt lõi của Trí tuệ nhân tạo, giúp giải quyết bài toán quá tải thông tin và cá nhân hóa trải nghiệm người dùng. Dự án mở rộng an toàn từ CineAI V2.0 thành nền tảng đa miền hoàn chỉnh, gồm **16 mô hình AI được huấn luyện và kiểm định khoa học**:

1. **🎬 CineAI (Phim ảnh — Explicit Feedback):**
   - Tập dữ liệu: **MovieLens 100K** (100.000 đánh giá 1–5 sao từ 943 người dùng trên 1.682 bộ phim).
   - **10 mô hình AI:** Neural CF (NeuMF), Weighted Hybrid, User-Based CF, Item-Based CF, KNN Baseline, SVD++, Funk SVD, Popularity Baseline, Non-negative MF (NMF), Content-Based (TF-IDF).
   - Giao diện Netflix-inspired chuẩn điện ảnh: Khám phá, Đề xuất cá nhân hóa, AI Lab & Benchmark, So sánh đối đầu (Model Battle).

2. **🎵 MusicAI (Âm nhạc — Implicit Feedback):**
   - Tập dữ liệu: **Million Song Dataset — Taste Profile** (MSD 10-core: 2.520 người dùng, 2.706 bài hát, 46.682 tương tác user–song).
   - **6 mô hình AI:** Item-Based CF (Top 1 Benchmark), User-Based CF, Implicit ALS (Hu-Koren-Volinsky), Bayesian Personalized Ranking (BPR-MF), Implicit Neural CF (PyTorch NeuMF), Popularity Baseline.
   - Gợi ý cá nhân hóa ở **cấp độ từng bài hát (`song_id`)**, loại trừ 100% bài hát đã nghe trong lịch sử huấn luyện (Deduplication), tự động chuyển sang Popularity khi gặp người dùng mới (Cold-Start Fallback).

3. **🔄 Domain Switcher (Chuyển đổi Ngữ cảnh Hai chiều):**
   - Nút chuyển đổi tức thời giữa *Movies* (theme đỏ Netflix) và *Music* (theme xanh ngọc lục bảo Emerald).
   - Tự động đồng bộ URL routing, menu điều hướng và bộ chọn người dùng mẫu theo miền.

---

## 2. Bảng Xếp hạng Benchmark Khoa học của 16 Mô hình AI

### A. CineAI: Bảng xếp hạng Top-10 trên Tập Test MovieLens 100K (N = 935 users)
| Hạng | Tên Mô hình | Họ Thuật toán | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 | Latency (ms) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | **Neural CF (NeuMF)** | **Deep Learning (PyTorch)** | **0.2251** | **0.2391** | **0.3083** | **82.03%** | 11.54 ms |
| **2** | **Hybrid Ensemble** | **Weighted Hybrid (CF + CB)** | **0.1475** | **0.1327** | **0.1943** | **65.78%** | 12.10 ms |
| 3 | User-Based CF | Neighborhood CF (Pearson KNN) | 0.1467 | 0.1352 | 0.1948 | 65.24% | 10.71 ms |
| 4 | Item-Based CF | Neighborhood CF (Adjusted Cosine) | 0.1247 | 0.1146 | 0.1611 | 59.04% | 10.85 ms |
| 5 | KNN Baseline | Neighborhood with User/Item Biases | 0.1059 | 0.0816 | 0.1316 | 50.27% | 3.53 ms |
| 6 | SVD++ | Matrix Factorization + Implicit History | 0.0896 | 0.0701 | 0.1089 | 47.59% | 4.67 ms |
| 7 | Funk SVD | Regularized Matrix Factorization | 0.0711 | 0.0508 | 0.0715 | 40.32% | 3.75 ms |
| 8 | Popularity Baseline | Bayesian Average Rating | 0.0653 | 0.0497 | 0.0764 | 39.14% | 2.15 ms |
| 9 | NMF | Non-Negative Matrix Factorization | 0.0542 | 0.0380 | 0.0525 | 33.16% | 3.32 ms |
| 10 | Content-Based | TF-IDF Cosine Similarity | 0.0236 | 0.0199 | 0.0296 | 18.72% | 5.37 ms |

### B. MusicAI: Bảng xếp hạng Top-10 trên Tập Test Million Song Dataset (N = 2.520 users)
| Hạng | Tên Mô hình | Họ Thuật toán | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 | Coverage | Latency (ms) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Item-Based CF** | **Item-Item Adjusted Cosine (Top-50)** | **0.0307** | **0.1544** | **0.1085** | **24.48%** | **93.20%** | **9.72 ms** |
| **2** | **User-Based CF** | **User Taste Cosine Vectors** | **0.0297** | **0.1535** | **0.1075** | **23.89%** | 65.45% | 3.15 ms |
| 3 | Implicit ALS (iALS) | Matrix Factorization (Confidence Log) | 0.0249 | 0.1377 | 0.0856 | 21.11% | 62.90% | 5.99 ms |
| 4 | BPR-MF | Pairwise Margin Ranking Loss | 0.0201 | 0.1066 | 0.0674 | 16.43% | 71.32% | 5.64 ms |
| 5 | Neural CF (NeuMF) | PyTorch Non-linear (GMF + MLP) | 0.0163 | 0.0886 | 0.0543 | 13.49% | 58.17% | 5.53 ms |
| 6 | Popularity Baseline | Global Play Count Frequency | 0.0125 | 0.0672 | 0.0398 | 10.04% | 3.70% | 2.62 ms |

---

## 3. Cấu trúc Thư mục Dự án (Project Layout)

```text
AI-Recommendation-System/
├── backend/                      # Dịch vụ FastAPI Backend hiệu năng cao
│   ├── app/
│   │   ├── api/v1/               # Router CineAI (/api/v1/*) & MusicAI (/api/music/*)
│   │   ├── core/                 # Cấu hình cài đặt, đường dẫn và CORS
│   │   ├── schemas/              # Khế ước dữ liệu Pydantic v2 (Movies & Music)
│   │   ├── services/             # Model Registry, Dịch vụ Phim, Dịch vụ Bài hát
│   │   └── main.py               # Điểm khởi chạy ASGI FastAPI Server
│   └── requirements.txt          # Dependencies tối giản cho Production Backend
├── frontend/                     # Ứng dụng React + TypeScript + Tailwind CSS
│   ├── src/
│   │   ├── components/           # UI Components (Navbar, DomainSwitcher, MovieCard, SongCard, Modals)
│   │   ├── context/              # UserContext (CineAI) & DomainContext (MusicAI & Switcher)
│   │   ├── pages/                # Route-based Lazy Loaded Pages (CineAI & MusicAI)
│   │   ├── services/             # api.ts (CineAI Client) & musicApi.ts (MusicAI Client)
│   │   └── test/                 # Vitest Component Unit Tests (11/11 tests PASS)
│   ├── e2e_multidomain.cjs       # Playwright E2E Browser Test Suite (22/22 scenarios PASS)
│   └── nginx.conf                # Nginx production reverse proxy & security headers
├── data/                         # Thư mục dữ liệu (Được quản lý tách biệt)
│   ├── processed/                # Dữ liệu CineAI đã làm sạch (movies_clean.csv, train.csv)
│   └── music/processed/          # Dữ liệu MusicAI đã tiền xử lý (songs.csv, train.csv)
├── models/checkpoints/           # Checkpoints CineAI (.pkl) & MusicAI (models/checkpoints/music/*.pkl)
├── docker-compose.yml            # Điều phối Container Docker (Backend + Frontend)
├── Dockerfile.backend            # Image Backend FastAPI (Python 3.11-slim, User appuser)
├── Dockerfile.frontend           # Image Frontend Multi-stage (Node 20 -> Nginx Unprivileged)
└── scripts/                      # Bộ công cụ kiểm toán, xác thực và đo đạc hiệu năng
```

---

## 4. Hướng dẫn Cài đặt và Chạy Hệ thống

### 4.1. Chuẩn bị Tiền điều kiện
- **Python:** 3.11+
- **Node.js:** v20+ hoặc v24+
- **Docker & Docker Compose:** (Nếu muốn chạy container hóa)

### 4.2. Clone Repository & Cài đặt Môi trường
```bash
# 1. Clone repository
git clone https://github.com/HoangTuan1301/AI-Recommendation-System.git
cd AI-Recommendation-System

# 2. Thiết lập Python Virtual Environment
python -m venv .venv
# Trên Windows:
.\.venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source .venv/bin/activate

# 3. Cài đặt Python Dependencies
pip install -r requirements.txt

# 4. Cài đặt Frontend Dependencies
cd frontend
npm install
cd ..
```

### 4.3. Xác thực Dữ liệu và Checkpoints
> [!IMPORTANT]
> **Tự động vs Cung cấp thủ công:**
> - Checkpoints và Dữ liệu đã xử lý được lưu trữ tại `models/checkpoints/` và `data/` (bị Git bỏ qua theo chuẩn ML).
> - Trước khi khởi động, chạy script kiểm tra toàn vẹn để đảm bảo 16 mô hình và các tệp dữ liệu đã sẵn sàng:

```bash
python scripts/verify_production_artifacts.py
```
*Kết quả hợp lệ:* `OVERALL ARTIFACT STATUS: ALL VERIFIED (PASS)` (16/16 checkpoints OK, 8/8 datasets OK).

---

## 5. Khởi động Ứng dụng

### Cách 1: Khởi động Trực tiếp (Local Development)

**Cửa sổ dòng lệnh 1 — Khởi động FastAPI Backend (Port 8000):**
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- Tài liệu tương tác Swagger UI: `http://127.0.0.1:8000/docs`
- Kiểm tra sức khỏe hệ thống: `http://127.0.0.1:8000/api/v1/health`

**Cửa sổ dòng lệnh 2 — Khởi động React Frontend (Port 5173):**
```bash
cd frontend
npm run dev
```
- Mở trình duyệt tại: `http://localhost:5173`

---

### Cách 2: Khởi động bằng Docker Compose (Khuyến nghị cho Production)

```bash
# Xây dựng và khởi chạy 2 containers ở chế độ nền
docker compose up -d --build

# Kiểm tra trạng thái sức khỏe
docker compose ps
```
- **Web App (Nginx SPA + Reverse Proxy):** `http://localhost:3000`
- **FastAPI Backend:** `http://localhost:8000/docs`

---

## 6. Hướng dẫn Kiểm thử Toàn bộ Hệ thống (Testing Suite)

Dự án sở hữu quy trình kiểm định chất lượng phần mềm nghiêm ngặt với 4 tầng kiểm thử:

### A. Kiểm thử Pytest Backend (134 Tests — 100% PASS)
Kiểm thử toàn vẹn hợp đồng API, thuật toán 16 mô hình, lazy loading, và không rò rỉ dữ liệu:
```bash
pytest tests/ -q
```
*Kết quả:* `134 passed in ~35s`.

### B. Kiểm thử Vitest Frontend (11 Tests — 100% PASS)
Kiểm thử giao diện CineAI, Domain Switcher, SongCard, MusicModelSelector, và Leaderboard:
```bash
cd frontend
npx vitest run
```
*Kết quả:* `11 passed (2 test files) in 1.2s`.

### C. Kiểm thử E2E Thực tế trên Trình duyệt bằng Playwright (22 Scenarios — 100% PASS)
Thực hiện 15 kịch bản tương tác người dùng thực tế với backend và frontend thật:
```bash
cd frontend
node e2e_multidomain.cjs
```
*Kết quả:* `22 passed, 0 failed, 0 console errors, 0 network errors`. Ảnh chụp màn hình lưu tại `reports/screenshots/e2e_m6/`.

### D. Đo đạc Hiệu năng & Bộ nhớ Thực tế
```bash
python scripts/benchmark_m6_performance.py
```
*Kết quả:* Thời gian đáp ứng API P95 < 15ms; Bộ nhớ RAM tiêu thụ 61.5 MB RSS.

---

## 7. Giấy phép & Thông tin Đóng góp
Dự án được nghiên cứu và phát triển trong khuôn khổ môn học Trí tuệ nhân tạo (Học kỳ 5). Toàn bộ mã nguồn tuân thủ giấy phép mã nguồn mở MIT.
