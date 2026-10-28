# BÁO CÁO KIỂM TOÁN HỆ THỐNG HIỆN HỮU (CINEAI V2.0)
## PHASE M0: REPOSITORY AUDIT & REUSE FEASIBILITY STUDY

**Dự án:** Multi-Domain AI Recommendation System (CineAI & MusicAI)  
**Nhánh Git:** `feature/musicai-foundation`  
**Trạng thái kiểm toán:** HOÀN THÀNH (AUDITED & CERTIFIED)  
**Tác giả:** Senior AI/ML Engineer & Recommender Systems Architect  
**Ngày thực hiện:** 28/10/2026  

---

## 1. TỔNG QUAN VÀ MỤC ĐÍCH KIỂM TOÁN

Hệ thống **CineAI V2.0** (Gợi ý phim thông minh trên MovieLens 100K) đã hoàn thành toàn diện giai đoạn Phase 6A với kết luận **GO ACCEPTANCE**:
- 10 mô hình học máy và học sâu đạt chuẩn benchmark khoa học.
- Toàn bộ 10 checkpoint mô hình được huấn luyện và lưu trữ hoàn chỉnh.
- Backend FastAPI (port 8000) và Frontend React 19 + TypeScript + Tailwind CSS (port 5173) hoạt động đồng bộ, ổn định.
- Bộ kiểm thử tự động 65+ unit/integration tests vượt qua 100%.

Mục đích của đợt kiểm toán Phase M0 là rà soát toàn bộ cấu trúc mã nguồn, dữ liệu, mô hình, API và giao diện của CineAI nhằm:
1. Xác định chính xác các thành phần nền tảng có thể tái sử dụng an toàn cho miền **MusicAI**.
2. Nhận diện các điểm khớp nối cứng (hard-coded couplings) về định dạng dữ liệu (phim vs bài hát, đánh giá sao tường minh vs tương tác ngầm định).
3. Đề xuất chiến lược mở rộng đa miền (Multi-Domain Architecture) độc lập, bảo đảm **nguyên tắc bất khả xâm phạm (zero-regression)** đối với CineAI.

---

## 2. KIỂM TOÁN DANH MỤC THÀNH PHẦN HIỆN CÓ CỦA CINEAI

### 2.1. Danh mục 10 Mô hình và Checkpoints (`models/checkpoints/`)

Toàn bộ 10 mô hình CineAI được tuần tự hóa (serialization) bằng `joblib` hoặc `torch` trong thư mục `models/checkpoints/`:

| STT | Tên mô hình | File Checkpoint | Kích thước | Họ thuật toán (Family) | Phương thức lưu trữ |
|---|---|---|---|---|---|
| 1 | Popularity Baseline | `popularity.pkl` | 324.6 KB | Baseline / Heuristic | `joblib` (Bayesian IMDB formula) |
| 2 | Content-Based Recommender | `content_based.pkl` | 25.6 MB | Content-Based / NLP | `joblib` (TF-IDF 2,350 chiều + Cosine) |
| 3 | User-Based Collaborative Filtering | `user_cf.pkl` | 5.2 MB | Memory-Based CF | `joblib` (Pearson correlation matrix) |
| 4 | Item-Based Collaborative Filtering | `item_cf.pkl` | 2.4 MB | Memory-Based CF | `joblib` (Adjusted Cosine sim matrix) |
| 5 | KNN with Baselines (KNNBaseline) | `knn_baseline.pkl` | 2.4 MB | Memory-Based CF + Bias | `joblib` (Shrinkage Cosine on residuals) |
| 6 | Funk SVD (Matrix Factorization) | `svd.pkl` | 596.3 KB | Latent Factor Model | `joblib` (User/Item embeddings k=20) |
| 7 | SVD++ | `svdpp.pkl` | 806.4 KB | Latent Factor + Implicit | `joblib` (Implicit feedback vector $y_j$) |
| 8 | Non-Negative Matrix Factorization | `nmf.pkl` | 911.3 KB | Latent Factor Model | `joblib` (Non-negative W, H matrices) |
| 9 | Neural Collaborative Filtering (NeuMF) | `neural_cf.pkl` | 1.5 MB | Deep Learning (PyTorch) | `joblib` (PyTorch GMF + MLP weights) |
| 10 | Weighted Hybrid Recommender | `hybrid.pkl` | 31.0 MB | Hybrid Ensemble | `joblib` (Ensemble SVD++ & Content-Based) |

> **Nhận xét kiểm toán:** Toàn bộ 10 file checkpoint đều nguyên vẹn, có dung lượng khớp chính xác với báo cáo Phase 4C và Phase 6A. Không có bất kỳ dấu hiệu lỗi serialize hoặc sai lệch kích thước.

---

### 2.2. Lớp giao tiếp cơ sở (`src/models/base_recommender.py`)

Giao diện trừu tượng `BaseRecommender` định nghĩa hợp đồng chung cho mọi mô hình:
```python
class BaseRecommender(ABC):
    def __init__(self, name: Optional[str] = None) -> None:
        self.name: str = name or self.__class__.__name__
        self.is_fitted: bool = False
        self.user_train_history: Dict[int, Set[int]] = {}
        self.all_train_items: Set[int] = set()
        self.all_train_users: Set[int] = set()

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseRecommender": ...

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        candidate_items: Optional[Sequence[int]] = None
    ) -> List[Tuple[int, float]]: ...

    def predict(self, user_id: int, item_id: int) -> float: ...
```

**Khảo sát tính tái sử dụng:**
- **Ưu điểm:** Phương thức `recommend(user_id, top_n, candidate_items)` trả về `List[Tuple[item_id, score]]` có tính tổng quát cao, hoàn toàn tương thích với cả phim (`movie_id`) và âm nhạc (`track_id` hoặc `artist_id`).
- **Điểm gắn kết (Coupling):** Docstring và một số phương thức giả định `train_df` chứa các cột cụ thể `['user_id', 'movie_id', 'rating']`. Cần thiết kế một base interface mở rộng hoặc cho phép cấu hình tên cột linh hoạt (`user_col`, `item_col`, `feedback_col`).

---

### 2.3. Đường ống dữ liệu (`data/processed/`)

Dữ liệu CineAI được cố định từ MovieLens 100K:
- `movies_clean.csv` (271.1 KB): Danh mục 1,682 phim kèm tiêu đề, năm phát hành, 19 thể loại nhị phân và nhãn ghép.
- `train.csv` (1.98 MB, 80,000 dòng): Tập huấn luyện 80% phân bổ phân tầng theo từng người dùng (seed=42).
- `test.csv` (495.5 KB, 20,000 dòng): Tập kiểm thử 20% giữ nguyên để đánh giá benchmark khách quan.
- `train_inner.csv` (1.59 MB, 64,000 dòng) và `val.csv` (396.2 KB, 16,000 dòng): Phân hoạch dùng cho siêu tham số hóa (hyperparameter tuning).
- `split_metadata.json` và `split_inner_metadata.json`: Bản ghi chứng thực tính toàn vẹn của phân hoạch.

> **Ranh giới an toàn:** Dữ liệu âm nhạc cho MusicAI tuyệt đối không được ghi đè hay trộn lẫn vào thư mục `data/processed/` của CineAI. MusicAI sẽ sử dụng không gian thư mục chuyên biệt `data/music/processed/` và `data/music/raw/`.

---

### 2.4. Khung đánh giá Top-N (`src/evaluation/evaluator.py`)

Bộ đánh giá chuẩn hóa `TopNEvaluator`:
- **Giao thức:** Đánh giá Top-K (mặc định $K=10$) trên toàn bộ 935 người dùng kiểm thử có ít nhất 1 tương tác tích cực.
- **Ngưỡng phù hợp (Relevance Threshold):** $r \ge 4.0$ (phản hồi tường minh trên thang điểm 1-5 sao).
- **Bộ ứng viên:** Tập mục khả dụng trong catalog trừ đi lịch sử xem trong tập huấn luyện của người dùng (`mask seen items`).
- **Độ đo triển khai:** Precision@K, Recall@K, NDCG@K, Hit Rate@K, F1@K (tính toán bình quân vĩ mô Macro-Average).

**Đánh giá tương thích cho MusicAI:**
- Trong miền âm nhạc, dữ liệu phản hồi thường là **ngầm định (implicit feedback)**: số lượt nghe (`play_count`, `listen_count`), số lần thêm vào danh sách (`added_to_playlist`), hoặc hành vi bấm thích (`like`). Do đó, khái niệm ngưỡng $r \ge 4.0$ không áp dụng trực tiếp được mà cần chuyển sang ngưỡng tương tác nhị phân ($c_{ui} > 0$) kèm chiến lược lấy mẫu phủ định (Negative Sampling).

---

### 2.5. Backend API (`backend/app/`)

Cấu trúc tầng dịch vụ và router của FastAPI:
- `backend/app/main.py`: Khởi tạo ứng dụng FastAPI, middleware CORS, tiền tố router `/api/v1`.
- Router con:
  - `/api/v1/health`: Kiểm tra sức khỏe dịch vụ.
  - `/api/v1/models`: Danh sách thông tin 10 mô hình AI.
  - `/api/v1/movies`: Tra cứu, lọc theo thể loại, phân trang danh mục phim.
  - `/api/v1/users`: Danh sách ID người dùng và lịch sử đánh giá.
  - `/api/v1/recommendations`: Sinh Top-N gợi ý phim theo mô hình chỉ định.
  - `/api/v1/metrics`: Bảng so sánh 5 độ đo benchmark từ `results/phase4c_metrics.csv`.
  - `/api/v1/compare`: So sánh trực tiếp danh sách gợi ý giữa hai mô hình bất kỳ.
- Tầng dịch vụ (`backend/app/services/`):
  - `model_service.py`: Quản lý nạp lazy-load các model checkpoints, ánh xạ ID và sinh dự đoán.
  - `movie_service.py`: Cache dữ liệu `movies_clean.csv`.
  - `poster_service.py`: Sinh poster SVG động theo thể loại và bảng màu.
  - `metrics_service.py`: Đọc và phân phối chỉ số đo lường.

**Đánh giá mở rộng:**
- Các endpoint hiện tại gắn với ngữ nghĩa phim (`movies`, `movie_id`, `movies_clean.csv`).
- Để hỗ trợ MusicAI mà không phá vỡ hợp đồng API hiện có, cần bổ sung router namespace riêng: `/api/v1/music/*` (`tracks`, `artists`, `recommendations`, `metrics`, `compare`). Các router hiện hữu của CineAI giữ nguyên 100% để bảo đảm tính tương thích ngược (backward compatibility).

---

### 2.6. Frontend React 19 (`frontend/src/`)

Giao diện CineAI xây dựng theo kiến trúc SPA hiện đại:
- **Ngăn xếp công nghệ:** React 19, TypeScript, Vite, Tailwind CSS v4, Lucide React icons.
- **5 trang chức năng hoàn chỉnh:**
  - `HomePage.tsx`: Hero banner điện ảnh, hàng phim gợi ý theo các danh mục thịnh hành.
  - `DiscoverPage.tsx`: Bộ lọc phim đa tiêu chí (thể loại, năm phát hành, tìm kiếm tên).
  - `RecommendationsPage.tsx`: Tương tác người dùng chọn User ID, Model AI, số lượng $N$ để nhận gợi ý tức thì.
  - `AILabPage.tsx`: Bảng so sánh khoa học 10 mô hình AI và công cụ so sánh trực quan đối đầu (Model Comparison).
  - `ProfilePage.tsx`: Lịch sử xem phim, phân bổ điểm đánh giá của từng tài khoản người dùng.
- **Hệ thống Design System:** Dark mode điện ảnh Netflix (#141414, #E50914, kính mờ glassmorphism, typography chuẩn hóa).

---

## 3. PHÂN TÍCH ĐIỂM GẮN KẾT CỨNG (COUPLING) VÀ RỦI RO LAN TRUYỀN

Dưới đây là các điểm gắn kết chặt chẽ với bài toán phim cần được bóc tách khi đưa MusicAI vào hệ thống:

```
+-----------------------------------------------------------------------------------+
|                            CINEAI COUPLING ANALYSIS                               |
+-----------------------------------------------------------------------------------+
| 1. Column Identity: 'movie_id' vs 'track_id' / 'artist_id'                        |
| 2. Feedback Regime: Explicit (1-5 stars) vs Implicit (listen count / frequency)   |
| 3. Relevance Rule:  r >= 4.0 vs Implicit consumption (c_ui > 0, confidence score)|
| 4. Candidate Pool:  1,682 movies vs 10k-50k tracks/artists (higher sparsity)      |
| 5. Media & Assets:  Movie poster/genres vs Album art/artist/track duration/audio   |
| 6. Storage Target:  models/checkpoints/*.pkl (shared flat directory)              |
+-----------------------------------------------------------------------------------+
```

1. **Khác biệt về bản chất phản hồi (Explicit vs Implicit):**
   - CineAI xử lý phản hồi tường minh (1 đến 5 sao). Người dùng chủ động cho điểm các bộ phim đã xem.
   - MusicAI xử lý phản hồi ngầm định (số lượt nghe, số lần nghe lặp lại, thời lượng nghe). Không có điểm phạt rõ ràng khi người dùng không nghe một bài hát (unobserved items không đồng nghĩa với việc người dùng ghét bài hát đó).
2. **Khác biệt về độ thưa thớt (Matrix Sparsity):**
   - MovieLens 100K có 100,000 tương tác trên $943 \times 1,682$ mục (độ đầy ~6.3%, độ thưa 93.7%).
   - Tập dữ liệu âm nhạc (ví dụ Last.fm 2K hoặc 1K) thường có độ thưa lên tới 99.2% - 99.8%. Số lượng bài hát/nghệ sĩ lớn hơn nhiều lần số lượng phim.
3. **Khác biệt về hành vi tiêu dùng (Consumption Behavior):**
   - Phim thường chỉ được xem 1 lần; việc gợi ý lại phim đã xem ít có ý nghĩa ngoại trừ xem lại phim kinh điển.
   - Âm nhạc có tính lặp lại cực kỳ cao (repeat consumption). Người dùng nghe một bài hát hàng chục, hàng trăm lần.

---

## 4. MA TRẬN PHÂN LOẠI TÁI SỬ DỤNG VÀ PHÂN TÁCH (REUSE MATRIX)

| Thành phần hệ thống | Trạng thái xử lý | Giải pháp kỹ thuật triển khai |
|---|---|---|
| **Công thức đo lường Top-N** (NDCG, Recall, Precision, Hit Rate) | **Tái sử dụng trực tiếp 100%** | Giữ nguyên module `src/evaluation/metrics.py`. Các hàm toán học tính toán trên danh sách ID đã xếp hạng là hoàn toàn bất biến theo miền. |
| **Hạ tầng CORS, Error Handlers, Logging** | **Tái sử dụng trực tiếp 100%** | Tái sử dụng middleware và cấu trúc exception handler trong FastAPI. |
| **Design Tokens & Theme** (Tailwind v4, Typography, Card styles) | **Tái sử dụng trực tiếp 100%** | Sử dụng chung hệ thống CSS, màu sắc chủ đạo kết hợp thêm dải màu neon xanh/tím (Spotify/SoundCloud style) cho miền MusicAI. |
| **Giao diện BaseRecommender** | **Tổng quát hóa (Generalize)** | Giữ nguyên `BaseRecommender` cho CineAI; tạo `BaseImplicitRecommender` trong `src/music/models/base_music_recommender.py` tối ưu cho phản hồi ngầm định và negative sampling. |
| **Bộ đánh giá Top-N** (`TopNEvaluator`) | **Chuyên biệt hóa (Specialize)** | Tạo `MusicTopNEvaluator` kế thừa logic xếp hạng nhưng áp dụng Leave-K-out hoặc Temporal Split với chiến lược lấy mẫu âm (Unobserved Negative Sampling). |
| **Tầng lưu trữ Checkpoint** | **Phân tách hoàn toàn (Isolate)** | Giữ nguyên `models/checkpoints/` cho CineAI; thiết lập `models/checkpoints/music/` riêng biệt cho MusicAI. |
| **Đường dẫn Dữ liệu** | **Phân tách hoàn toàn (Isolate)** | Giữ nguyên `data/processed/` cho CineAI; thiết lập `data/music/` độc lập cho MusicAI. |
| **API Endpoints & Routers** | **Phân tách Namespace (Isolate)** | Giữ nguyên `/api/v1/movies/*`, `/api/v1/recommendations`; bổ sung `/api/v1/music/*` (`tracks`, `recommendations`, `metrics`). |
| **Trang Frontend & Kho lưu trữ trạng thái** | **Phân tách theo Route (Isolate)** | Bổ sung Switcher miền `[ CineAI | MusicAI ]` trên Navbar; các trang MusicAI hoạt động theo route `/music/*` độc lập. |

---

## 5. CAM KẾT VÀ CƠ CHẾ BẢO ĐẢM KHÔNG SUY THOÁI CINEAI (NON-REGRESSION GUARANTEE)

Nhằm bảo đảm sản phẩm CineAI V2.0 không chịu bất kỳ rủi ro suy thoái hay hỏng hóc nào:

1. **Bảo tồn Checkpoint Tuyệt đối:** Không ghi đè, di chuyển hay đổi tên bất kỳ file nào trong 10 checkpoint hiện hữu (`models/checkpoints/*.pkl`).
2. **Bảo tồn Dữ liệu & Benchmark:** Tập tin `data/processed/train.csv`, `test.csv`, `movies_clean.csv` và `results/phase4c_metrics.csv` được giữ nguyên vẹn 100%.
3. **Tính tương thích ngược API (Backward Compatibility):** Mọi client hoặc script kiểm thử đang gọi tới `/api/v1/movies`, `/api/v1/recommendations`, `/api/v1/metrics` đều tiếp tục nhận được kết quả phim chính xác như cũ mà không thay đổi định dạng schema.
4. **Cô lập Kiểm thử (Test Isolation):** Các bài kiểm thử hiện có của CineAI (`test_fastapi_backend.py`, `test_phase4c_audit.py`, `test_phase5_app.py`) phải luôn luôn pass 100% trong suốt quá trình phát triển MusicAI.

---

## 6. KẾT LUẬN KIỂM TOÁN PHASE M0

- **Hiện trạng CineAI:** Khỏe mạnh, cấu trúc module rõ ràng, hoàn toàn sẵn sàng đóng vai trò là cột trụ thứ nhất trong nền tảng Đa Miền (Multi-Domain Platform).
- **Tính khả thi mở rộng:** Rất cao. Các mẫu thiết kế (Design Patterns) của CineAI (Registry, Service Layer, Top-N Evaluation Harness) có thể nhân bản và chuyên biệt hóa sạch sẽ cho MusicAI.
- **Đánh giá rủi ro hồi quy:** 0% nếu tuân thủ nghiêm ngặt nguyên tắc phân tách không gian tên (namespace isolation) đã vạch ra.
