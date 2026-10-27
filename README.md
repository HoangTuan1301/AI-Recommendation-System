# AI-Recommendation-System (CineAI Platform)
> Hệ thống gợi ý phim thông minh ứng dụng Trí tuệ nhân tạo (AI Recommendation System) với giao diện Netflix-Inspired hiện đại.

Dự án Bài tập lớn môn **Trí tuệ nhân tạo (AI)** - Học kỳ 5.  
**Khoa Công nghệ Thông tin - Trường Đại học**  
**Tác giả:** Đội ngũ Kỹ sư AI & Khoa học Dữ liệu K5  
**GitHub:** [https://github.com/HoangTuan1301/AI-Recommendation-System](https://github.com/HoangTuan1301/AI-Recommendation-System)

---

## 1. Giới thiệu Đề tài & Nền tảng CineAI

Hệ thống khuyến nghị (Recommendation System) là phân nhánh quan trọng của Trí tuệ nhân tạo và Học máy, đóng vai trò cá nhân hóa trải nghiệm người dùng trên các nền tảng nội dung số (Netflix, Spotify, YouTube,...). Dự án nghiên cứu, thực nghiệm và so sánh toàn diện **10 mô hình AI gợi ý phim** trên bộ dữ liệu chuẩn mực quốc tế **MovieLens 100K**, được đánh giá theo **5 độ đo xếp hạng Top-N** và tích hợp trên hai nền tảng giao diện:

1. **CineAI Full-Stack Platform (Primary):** Giao diện Netflix-inspired chuẩn điện ảnh bằng **React 19 + TypeScript + Vite + Tailwind CSS**, kết nối với backend hiệu năng cao **FastAPI** phục vụ suy luận thời gian thực từ checkpoint các mô hình AI.
2. **Streamlit App (Fallback):** Giao diện Python thuần nhanh gọn (`app/streamlit_app.py`) phục vụ trình diễn và đối chiếu nhanh.

---

## 2. Bảng Xếp hạng Chính thức 10 Mô hình AI (Benchmark Leaderboard)

Đánh giá thực nghiệm độc lập trên tập Test (**935 người dùng kiểm thử**, loại trừ 100% phim đã xem trong Train):

| Hạng | Tên Mô hình | Họ Thuật toán | Precision@10 | Recall@10 | F1-Score@10 | NDCG@10 | Hit Rate@10 | Thời gian Eval |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Neural CF (NeuMF)** | **Deep Learning (GMF + MLP)** | **0.2251** | **0.2391** | **0.1898** | **0.3083** | **0.8203** | 2.27s |
| **2** | **Hybrid (CF + CB)** | **Weighted Hybrid Ensemble** | **0.1475** | **0.1327** | **0.1164** | **0.1943** | **0.6578** | 8.35s |
| 3 | User-Based CF | Memory-based CF (User KNN) | 0.1467 | 0.1352 | 0.1160 | 0.1948 | 0.6524 | 4.39s |
| 4 | Item-Based CF | Memory-based CF (Item KNN) | 0.1247 | 0.1146 | 0.0979 | 0.1611 | 0.5904 | 4.70s |
| 5 | KNNBaseline | Neighborhood CF with Biases | 0.1059 | 0.0816 | 0.0778 | 0.1316 | 0.5027 | 3.53s |
| 6 | SVD++ | Matrix Factorization + Implicit | 0.0896 | 0.0701 | 0.0671 | 0.1089 | 0.4759 | 4.67s |
| 7 | Funk SVD | Regularized Matrix Factorization | 0.0711 | 0.0508 | 0.0508 | 0.0715 | 0.4032 | 3.75s |
| 8 | Popularity Baseline | Bayesian Weighted Rating | 0.0653 | 0.0497 | 0.0473 | 0.0764 | 0.3914 | 3.23s |
| 9 | NMF | Non-Negative Matrix Factorization | 0.0542 | 0.0380 | 0.0387 | 0.0525 | 0.3316 | 3.32s |
| 10 | Content-Based | TF-IDF Cosine Similarity | 0.0236 | 0.0199 | 0.0174 | 0.0296 | 0.1872 | 5.37s |

> [!NOTE]
> **Giải thích Hit Rate@10 = 82.03%:** Nghĩa là trong 935 người dùng kiểm thử, có đúng **82.03% người dùng nhận được ít nhất một bộ phim thực sự yêu thích ($r \ge 4.0$) trong danh sách Top 10 gợi ý**.

---

## 3. Kiến trúc Toàn diện (System Architecture)

```text
AI-Recommendation-System/
├── backend/                      # High-performance FastAPI Backend Service
│   └── app/
│       ├── api/v1/               # REST API Routes: health, models, movies, users, recommendations, metrics, compare
│       ├── core/                 # App Settings & CORS configurations
│       ├── schemas/              # Pydantic v2 Request & Response Data Contracts
│       ├── services/             # Model Checkpoint Loader, User/Movie Catalog & SVG Poster Generator
│       └── main.py               # FastAPI ASGI Application Entrypoint
├── frontend/                     # Modern React + TypeScript + Tailwind CSS Frontend
│   ├── src/
│   │   ├── api/client.ts         # Axios/Fetch API client kết nối Backend
│   │   ├── components/           # UI Components (Navbar, Footer, HeroBanner, MovieCard, Carousel, Modals)
│   │   ├── context/UserContext   # State quản lý Active User ID (1-943) và User Modal
│   │   ├── pages/                # 5 Views: Home, Discover, Recommendations, AILab, Profile
│   │   └── test/                 # Vitest Component & Integration Tests (6/6 tests PASS)
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.js        # Netflix Dark Design Tokens (#08090D, #E50914, #181A21)
│   └── vite.config.ts            # Vite bundler với reverse proxy /api -> localhost:8000
├── app/                          # Streamlit UI (Fallback)
│   ├── model_loader.py           # Streamlit Checkpoint Loader
│   └── streamlit_app.py          # Streamlit Multi-page App
├── data/
│   ├── raw/ml-100k/              # Dữ liệu thô MovieLens 100K
│   └── processed/                # Dữ liệu sạch: train.csv, test.csv, movies_clean.csv
├── models/
│   └── checkpoints/              # Checkpoint đã serialize (.pkl) cho 10 mô hình
├── reports/                      # Báo cáo kỹ thuật chi tiết các giai đoạn
├── results/                      # Kết quả benchmark: phase4c_metrics.csv
├── src/                          # Core AI Recommendation Algorithms & Evaluation Framework
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── evaluation/               # Metrics, TopNEvaluator, Hyperparameter Tuning
│   └── models/                   # 10 giải thuật AI khuyến nghị độc lập
├── tests/                        # 85 bài kiểm thử tự động Python (100% PASS)
├── README.md
└── requirements.txt
```

---

## 4. Hướng dẫn Cài đặt & Khởi chạy

### Bước 1: Môi trường Python & Thư viện Backend
```powershell
# Di chuyển vào thư mục dự án
cd E:\HOCTAP\K5\TTNT\AI-Recommendation-System

# Tạo và kích hoạt môi trường ảo (nếu chưa có)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài đặt thư viện phụ thuộc
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 2: Khởi chạy FastAPI Backend (Port 8000)
Mở một cửa sổ Terminal 1:
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --reload --port 8000
```
- API Docs tự động (Swagger UI): `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/v1/health`

### Bước 3: Cài đặt và Khởi chạy React Frontend (Port 5173)
Mở một cửa sổ Terminal 2:
```powershell
cd frontend
npm install
npm run dev
```
Truy cập giao diện CineAI tại: `http://localhost:5173`

*(Tùy chọn) Chạy giao diện Streamlit Fallback (Port 8501):*
```powershell
python -m streamlit run app/streamlit_app.py
```

---

## 5. Trải nghiệm 5 Chức năng Chính trên CineAI

1. **🏠 Home Page (Trang chủ Điện ảnh):**
   - Hero banner động giới thiệu tác phẩm đề xuất chất lượng cao kèm tóm tắt và thể loại.
   - Các hàng phim băng chuyền (Carousels): *Top Rated Masterpieces*, *Trending on CineAI*, *Deep Learning AI Top Picks*.
   - Modal xem chi tiết phim với tỷ lệ đánh giá và các tác phẩm tương tự.

2. **🔍 Discover Page (Khám phá & Bộ lọc):**
   - Tìm kiếm phim theo tên theo thời gian thực (Debounced search).
   - Bộ lọc đa thể loại (Action, Sci-Fi, Drama, Romance,...).
   - Sắp xếp linh hoạt theo Đánh giá cao nhất, Đánh giá nhiều nhất, Phim mới nhất, hoặc Tên A-Z.

3. **🤖 AI Recommendations (Gợi ý Cá nhân hóa Đa Mô hình):**
   - Cho phép chọn bất kỳ mô hình nào trong **10 mô hình AI** thuộc 5 họ thuật toán.
   - Lựa chọn độ dài danh sách (Top 5, 10, 20).
   - Giải thích minh bạch ngữ nghĩa điểm số (Score Semantics) và độ phức tạp tính toán $O(\cdot)$.
   - Loại trừ 100% các phim người dùng đã xem trong tập Train.

4. **🔬 AI Lab & Benchmark (Phòng Thí nghiệm AI):**
   - Bảng xếp hạng chính thức (Leaderboard) sắp xếp theo từng độ đo Top-N.
   - Biểu đồ tương tác đa độ đo (Precision, Recall, F1, NDCG, Hit Rate).
   - **Model Battle (So găng 2 Mô hình):** So sánh trực tiếp danh sách Top-10 giữa 2 mô hình bất kỳ cho một người dùng, tự động tính hệ số tương đồng Jaccard và làm nổi bật các phim đồng thuận.
   - Giải thích chi tiết toán học và ý nghĩa các độ đo khuyến nghị.

5. **👤 User Profile (Hồ sơ Sở thích):**
   - Trình chuyển đổi người dùng nhanh (Quick User Switcher từ 1 đến 943).
   - Thống kê hồ sơ dựa trên lịch sử xem thực tế trong Train: Tổng số phim đã xem, điểm trung bình, phân bố sao và top thể loại ưa thích.

---

## 6. Kiểm thử Tự động & Đảm bảo Chất lượng (Quality Assurance)

Dự án sở hữu bộ kiểm thử tự động toàn diện từ Core AI, Backend REST API cho tới Frontend Component:

### A. Kiểm thử Python (85 Tests - 100% PASS)
Bao gồm kiểm tra không rò rỉ dữ liệu (No Data Leakage), tính chính xác toán học của 10 mô hình, tính toàn vẹn của tập dữ liệu và toàn bộ 9 REST endpoints của FastAPI:
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
*Kết quả:* **Ran 85 tests in ~21s — OK**.

### B. Kiểm thử Frontend (Vitest - 100% PASS)
Bao gồm kiểm tra Render Navbar, MovieCard, thẻ gợi ý với điểm số và rank, bảng chỉ số AI Lab, bộ lọc và context chuyển đổi người dùng:
```powershell
cd frontend
npm test -- --run
```
*Kết quả:* **6 / 6 tests PASSED**.

### C. Build Production Frontend
```powershell
cd frontend
npm run build
```
*Kết quả:* Biên dịch TypeScript và đóng gói Vite thành công (`dist/` tối ưu hoá sẵn sàng triển khai).

---

## 7. Giấy phép & Thông tin Đóng góp
Dự án được phát triển phục vụ mục đích nghiên cứu, học tập trong khuôn khổ môn học Trí tuệ nhân tạo (Học kỳ 5). Mọi mã nguồn tuân thủ giấy phép mã nguồn mở MIT.
