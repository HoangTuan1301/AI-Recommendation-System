# MUSICAI — PHASE M5: FRONTEND ARCHITECTURE SPECIFICATION

**Tài liệu:** Báo cáo Kiến trúc Giao diện Đa miền (Multi-Domain Frontend Architecture)  
**Phiên bản:** 1.0  
**Giai đoạn:** Phase M5  
**Hệ thống:** CineAI V2.0 + MusicAI  
**Công nghệ:** React 18, TypeScript, Tailwind CSS, Lucide Icons, Vite  

---

## 1. TỔNG QUAN KIẾN TRÚC FRONTEND ĐA MIỀN

Nhằm bảo tồn 100% tính nguyên vẹn và ổn định của **CineAI V2.0** (Giao diện chuẩn Netflix-style dành cho gợi ý phim), Phase M5 áp dụng kiến trúc **Context-Driven Domain Switching**. Cả hai miền trải nghiệm (Movies và Music) chia sẻ cùng một single-page application (SPA), cùng cơ sở hạ tầng layout cốt lõi, nhưng được cách ly hoàn toàn về state, hooks, API client và routing.

```
                           +---------------------------+
                           |     React Application     |
                           |   (App.tsx / main.tsx)    |
                           +-------------+-------------+
                                         |
                       +-----------------+-----------------+
                       |                                   |
           +-----------v-----------+           +-----------v-----------+
           |      UserProvider     |           |     DomainProvider    |
           |   (CineAI User State) |           |  (Domain & Music State)
           +-----------+-----------+           +-----------+-----------+
                       |                                   |
                       +-----------------+-----------------+
                                         |
                           +-------------v-------------+
                           |       Unified Layout      |
                           |  (Navbar + DomainSwitcher)|
                           +-------------+-------------+
                                         |
                 +-----------------------+-----------------------+
                 |                                               |
     +-----------v-----------+                       +-----------v-----------+
     |     CineAI Routes     |                       |     MusicAI Routes    |
     | - / (Discover)        |                       | - /music (Home)       |
     | - /recommendations    |                       | - /music/recommend    |
     | - /lab (Benchmark)    |                       | - /music/models       |
     | - /compare            |                       +-----------+-----------+
     +-----------+-----------+                                   |
                 |                                   +-----------v-----------+
     +-----------v-----------+                       |    Music Components   |
     |   CineAI Components   |                       | - SongCard            |
     | - MovieCard           |                       | - MusicModelSelector  |
     | - ModelSelector (10)  |                       | - SongDetailModal     |
     | - MetricsExplainer    |                       | - UserHistoryModal    |
     +-----------+-----------+                       +-----------+-----------+
                 |                                               |
     +-----------v-----------+                       +-----------v-----------+
     |   cineApi Service     |                       |    musicApi Service   |
     |   (/api/v1/...)       |                       |    (/api/music/...)   |
     +-----------------------+                       +-----------------------+
```

---

## 2. DOMAIN CONTEXT VÀ QUẢN LÝ TRẠNG THÁI (STATE MANAGEMENT)

### 2.1. DomainContext (`src/context/DomainContext.tsx`)
`DomainContext` là hạt nhân điều phối việc chuyển đổi giữa hai hệ thống:
- Quản lý trạng thái miền kích hoạt: `activeDomain: 'movies' | 'music'`.
- Tự động đồng bộ với URL router: khi truy cập các route `/music*`, domain tự động chuyển sang `music`; khi truy cập các route gốc (`/`, `/recommendations`, `/lab`, `/compare`), domain tự động chuyển sang `movies`.
- Quản lý định danh người dùng MusicAI: `musicUserId: string` (mặc định khởi tạo `b80344d063b5ccb3212f76538f3d9e43d87dca9e`).
- Cung cấp danh sách người dùng mẫu MSD đại diện đa dạng gu âm nhạc:
  1. `b80344d063b5ccb3212f76538f3d9e43d87dca9e` (Demo Listener A - Diverse Pop/Rock)
  2. `85c1f87fea955d09b4bec2e36aee110927aedd33` (Demo Listener B - Rock & Indie)
  3. `bd4c6e843f00bd4773848b659b8b07f3ba779737` (Demo Listener C - Electronic & Synth)
  4. `9f8e43e6205776d63bb1c35bba6512bb3d9cc2ab` (Demo Listener D - Pop & Soul)
  5. `1111111111111111111111111111111111111111` (Cold-Start Simulated User)
- Quản lý trạng thái Modals toàn cục: `selectedSongId` (xem chi tiết bài hát) và `historyModalOpen` (xem lịch sử huấn luyện).

### 2.2. DomainSwitcher (`src/components/layout/DomainSwitcher.tsx`)
- Hiển thị dạng thanh trượt đa năng (dual-pill toggle) hiện đại:
  - **Movies**: biểu tượng `Film`, màu đỏ Netflix (`#E50914`).
  - **Music**: biểu tượng `Music`, màu xanh ngọc lục bảo Emerald/Teal (`#10B981`).
- Khi click, switcher cập nhật `activeDomain` và kích hoạt điều hướng router:
  - Chuyển sang Music: điều hướng tới `/music/recommendations`.
  - Chuyển sang Movies: điều hướng tới `/recommendations`.

---

## 3. CÁC THÀNH PHẦN GIAO DIỆN MUSICAI (COMPONENTS)

### 3.1. SongCard (`src/components/music/SongCard.tsx`)
- Thẻ bài hát chuẩn responsive, hiển thị đĩa vinyl hoạt họa quay khi hover.
- Background gradient sinh theo hash của `song_id`, tạo sự sinh động trực quan không cần ảnh bìa tĩnh.
- Huy hiệu thứ hạng (Rank Badge) với màu sắc phân cấp:
  - Top 1: Vàng kim (Amber/Gold).
  - Top 2: Bạc (Zinc/Silver).
  - Top 3: Đồng (Bronze/Amber-700).
  - Top 4+: Xám đậm (Zinc-800).
- Điểm tương đồng/dự đoán (Affinity Score) hiển thị dạng badge ngọc lục bảo kèm biểu tượng sao lấp lánh (`Sparkles`).
- Mã nhận diện Million Song Dataset: `song_id` và `track_id`.
- Nút "Song Details" mở Modal chi tiết bài hát.

### 3.2. MusicModelSelector (`src/components/music/MusicModelSelector.tsx`)
- Hiển thị dạng thẻ lưới (Grid Card) cho cả 6 mô hình AI:
  1. **Item-Based CF** (Top 1 Benchmark, Mặc định, adjusted cosine).
  2. **User-Based CF** (Runner-up, cosine similarity giữa user vectors).
  3. **Implicit ALS** (Hu-Koren-Volinsky matrix factorization, 32-dim latent space).
  4. **BPR-MF** (Bayesian Personalized Ranking pairwise loss).
  5. **Neural CF (NeuMF)** (PyTorch Pyramidal MLP + Generalized Matrix Factorization).
  6. **Popularity Baseline** (Tần suất phát và số lượng người nghe tích lũy).
- Mỗi thẻ mô hình hiển thị:
  - Tên mô hình và họ thuật toán (`Neighborhood CF`, `Matrix Factorization`, `Deep Learning`, `Baseline`).
  - Huy hiệu "DEFAULT" nổi bật cho Item-CF.
  - Profile thời gian đáp ứng (Latency profile) thực tế.
  - Mô tả nguyên lý hoạt động ngắn gọn.

### 3.3. SongDetailModal (`src/components/music/SongDetailModal.tsx`)
- Hiển thị thông tin siêu dữ liệu bài hát lấy từ endpoint `/api/music/songs/{song_id}`:
  - Tên bài hát (`title`), Nghệ sĩ (`artist_name`).
  - Mã bài hát MSD (`song_id`), Mã track Echo Nest (`track_id`).
  - Thống kê toàn tập dữ liệu huấn luyện: Tổng số lượt nghe (`total_play_count`), Số lượng người nghe duy nhất (`unique_listeners`).
  - Kiểm tra tính hợp lệ và ánh xạ 1:1 chuẩn xác giữa `song_id` và `track_id`.

### 3.4. MusicUserHistoryModal (`src/components/music/MusicUserHistoryModal.tsx`)
- Kết nối tới endpoint `/api/music/users/{user_id}/history`.
- Hiển thị toàn bộ lịch sử nghe đã ghi nhận trong tập huấn luyện `train.csv` của người dùng.
- Giúp người kiểm thử và người dùng đối chiếu ground truth, xác minh tính cá nhân hóa và bảo đảm không có bài hát nào trong lịch sử xuất hiện lại trong danh sách gợi ý (Deduplication validation).

---

## 4. ĐỊỀU HƯỚNG VÀ CÁC TRANG CHỨC NĂNG (PAGES & ROUTING)

| Đường dẫn (URL) | Trang thành phần | Mô tả chức năng |
|:---|:---|:---|
| `/music` | `MusicHomePage` | Trang chủ âm nhạc: Hero banner, chỉ số catalog live (`/api/music/health`), giới thiệu 6 mô hình và liên kết nhanh. |
| `/music/recommendations` | `MusicRecommendationsPage` | Bảng điều khiển gợi ý cá nhân hóa: Chọn người dùng mẫu / nhập User ID bất kỳ, chọn mô hình AI, chọn Top-K (5, 10, 20), hiển thị danh sách bài hát gợi ý và cảnh báo Fallback nếu gặp người dùng Cold-start. |
| `/music/models` | `MusicModelsPage` | Bảng xếp hạng và benchmark khoa học chính thức (Offline Evaluation Benchmark từ Phase M3) trên 2.520 người dùng test split. |

---

## 5. BẢO TOÀN NGUYÊN VẸN CINEAI V2.0

- Các trang CineAI (`/`, `/recommendations`, `/lab`, `/compare`) được giữ nguyên toàn bộ logic, state và style.
- `UserContext` của CineAI tiếp tục lưu trữ người dùng MovieLens và danh sách mô hình CineAI độc lập.
- Giao diện CineAI không bị ảnh hưởng bởi các dependency hay styles của MusicAI, bảo đảm tính hồi quy bằng 0 (Zero Regression).
