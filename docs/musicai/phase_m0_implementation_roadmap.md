# LỘ TRÌNH TRIỂN KHAI NỀN TẢNG ĐA MIỀN (MUSICAI ROADMAP)
## PHASE M0: END-TO-END IMPLEMENTATION ROADMAP (PHASES M0 – M6)

**Dự án:** Multi-Domain AI Recommendation Platform  
**Tài liệu:** Kế hoạch hành động chi tiết từ Giai đoạn M0 đến M6  
**Trạng thái:** HOÀN THIỆN & ĐƯỢC PHÊ DUYỆT KIẾN TRÚC  
**Tác giả:** Senior Full-Stack Software Architect & Technical Lead  
**Ngày:** 28/10/2026  

---

## 1. TỔNG QUAN CÁC GIAI ĐOẠN PHÁT TRIỂN (PHASE OVERVIEW)

Quá trình mở rộng sang **MusicAI** được chia thành 7 giai đoạn nối tiếp có kiểm soát rủi ro nghiêm ngặt (Gated Milestones), bảo đảm mỗi bước đều có bài kiểm thử tự động xác thực và không ảnh hưởng đến CineAI:

```
[ Phase M0 ] Foundation & Architecture Audit (HIỆN TẠI - GO DECISION)
     │
     ▼
[ Phase M1 ] Music Dataset Acquisition, Ingestion, Cleaning & EDA
     │
     ▼
[ Phase M2 ] MusicAI Recommendation Engine Suite (6 Models)
     │
     ▼
[ Phase M3 ] Top-N Implicit Evaluation, Hyperparameter Tuning & Benchmark Report
     │
     ▼
[ Phase M4 ] FastAPI Backend Expansion (/api/v1/music/* & Services)
     │
     ▼
[ Phase M5 ] React 19 Frontend Expansion (Domain Switcher, Music Views, Audio Bar)
     │
     ▼
[ Phase M6 ] Multi-Domain System Acceptance, Integration Audit & Final Defense Prep
```

---

## 2. KẾ HOẠCH CHI TIẾT TỪNG GIAI ĐOẠN

---

### GIAI ĐOẠN M0: KIỂM TOÁN NỀN TẢNG VÀ THIẾT KẾ KIẾN TRÚC (CURRENT PHASE)
- **Mục tiêu:** Rà soát mã nguồn CineAI, nghiên cứu tính khả thi dữ liệu âm nhạc, thiết kế kiến trúc phân tách đa miền và xây dựng lộ trình thực thi.
- **Đầu vào:** Mã nguồn CineAI V2.0, các checkpoint hiện hữu, bộ kiểm thử 65+ tests.
- **Hoạt động thực hiện:**
  - Kiểm toán cấu trúc 10 mô hình và checkpoint của CineAI.
  - Phân tích so sánh 4 tập dữ liệu âm nhạc (Last.fm 2K, Last.fm 1K, MSD, Spotify).
  - Soạn thảo 6 tài liệu kiến trúc kỹ thuật trong `docs/musicai/`.
- **Sản phẩm bàn giao:**
  1. `docs/musicai/phase_m0_repository_audit.md`
  2. `docs/musicai/phase_m0_dataset_selection.md`
  3. `docs/musicai/phase_m0_architecture.md`
  4. `docs/musicai/phase_m0_model_strategy.md`
  5. `docs/musicai/phase_m0_implementation_roadmap.md`
  6. `docs/musicai/phase_m0_completion_report.md`
- **Tiêu chí nghiệm thu (Acceptance Criteria):**
  - Không có bất kỳ dòng mã nào của CineAI bị thay đổi ngoài ý muốn.
  - Toàn bộ 65+ bài kiểm thử CineAI tiếp tục vượt qua 100%.
  - Bản báo cáo kết luận đạt trạng thái **GO** cho Phase M1.

---

### GIAI ĐOẠN M1: THU THẬP, TIỀN XỬ LÝ DỮ LIỆU ÂM NHẠC VÀ EDA
- **Mục tiêu:** Tải, làm sạch và phân hoạch tập dữ liệu Last.fm 2K (HetRec 2011) thành cấu trúc chuẩn hoá phục vụ huấn luyện và đánh giá.
- **Đầu vào:** Bộ dữ liệu Last.fm 2K HetRec 2011 (tệp `user_artists.dat`, `artists.dat`, `tags.dat`, `user_taggedartists.dat`).
- **Hoạt động thực hiện:**
  - Viết module nạp và làm sạch dữ liệu: `src/music/data/data_loader.py`.
  - Loại bỏ các nghệ sĩ/người dùng có số lượng tương tác quá ít ($k$-core filtering với $k=5$).
  - Phân hoạch dữ liệu theo từng người dùng (User-Stratified Split, `seed=42`):
    - `data/music/processed/train.csv` (80%)
    - `data/music/processed/val.csv` (10%)
    - `data/music/processed/test.csv` (10%)
    - `data/music/processed/artists_clean.csv` (Metadata nghệ sĩ và tag đại diện).
  - Thực hiện Exploratory Data Analysis (EDA): vẽ đồ thị phân bổ lượt nghe (Long-tail distribution), đo lường độ thưa thớt, thống kê top thể loại và nghệ sĩ.
- **Sản phẩm bàn giao:**
  - Thư mục `data/music/processed/` với các tệp CSV chuẩn hóa.
  - Báo cáo phân tích dữ liệu: `docs/musicai/phase_m1_data_eda_report.md`.
  - Bộ kiểm thử dữ liệu: `tests/test_music_data_pipeline.py`.
- **Rào chắn CineAI:** Không tác động vào `data/processed/train.csv` hay `movies_clean.csv`.

---

### GIAI ĐOẠN M2: XÂY DỰNG KHUNG MÔ HÌNH HỌC MÁY MUSICAI (6 MODELS)
- **Mục tiêu:** Cài đặt 6 thuật toán gợi ý phản hồi ngầm định theo đúng chuẩn toán học và hướng đối tượng.
- **Đầu vào:** `data/music/processed/train.csv` và `val.csv`.
- **Hoạt động thực hiện:**
  - Xây dựng lớp trừu tượng `BaseMusicRecommender` trong `src/music/models/base_music_recommender.py`.
  - Cài đặt 6 mô hình:
    1. `PopularityMusicRecommender` (Global play-count / unique listeners).
    2. `ItemCFMusicRecommender` (Cosine / Jaccard similarity trên co-listening matrix).
    3. `UserCFMusicRecommender` (Neighborhood user-user implicit similarity).
    4. `IALSMusicRecommender` (Implicit Alternating Least Squares - Hu, Koren & Volinsky).
    5. `BPRMFMusicRecommender` (Bayesian Personalized Ranking Matrix Factorization qua SGD).
    6. `ImplicitNeuralCFRecommender` (NeuMF kết hợp GMF + MLP trên PyTorch với BCE loss).
  - Tối ưu hóa siêu tham số trên tập `val.csv`.
  - Huấn luyện và tuần tự hóa lưu checkpoint tại `models/checkpoints/music/*.pkl`.
- **Sản phẩm bàn giao:**
  - Mã nguồn 6 mô hình trong `src/music/models/`.
  - 6 tệp checkpoint trong `models/checkpoints/music/`.
  - Bộ kiểm thử mô hình: `tests/test_music_models.py`.
- **Rào chắn CineAI:** Không đụng chạm vào 10 file checkpoint trong `models/checkpoints/*.pkl`.

---

### GIAI ĐOẠN M3: ĐÁNH GIÁ TOP-N KHÁCH QUAN VÀ BÁO CÁO BENCHMARK
- **Mục tiêu:** Đánh giá khách quan 6 mô hình MusicAI trên tập kiểm thử độc lập `test.csv` và tạo bảng so sánh khoa học.
- **Đầu vào:** 6 checkpoints của MusicAI, `train.csv`, `test.csv`.
- **Hoạt động thực hiện:**
  - Xây dựng harness đánh giá: `src/music/evaluation/music_evaluator.py`.
  - Tính toán 5 chỉ số chuẩn cho từng mô hình: HR@10, Precision@10, Recall@10, NDCG@10, MRR@10.
  - Phân tích hiệu năng đối đầu: So sánh mô hình phân rã ma trận (iALS, BPR) với mô hình học sâu (NeuMF) và các phương pháp lân cận (Item-CF, User-CF).
  - Xuất bảng kết quả ra file `results/music_metrics.csv`.
- **Sản phẩm bàn giao:**
  - Tập tin `results/music_metrics.csv`.
  - Báo cáo phân tích chuyên sâu: `docs/musicai/phase_m3_benchmark_report.md`.
  - Bộ kiểm thử đánh giá: `tests/test_music_evaluator.py`.
- **Rào chắn CineAI:** Giữ nguyên `results/phase4c_metrics.csv` của CineAI.

---

### GIAI ĐOẠN M4: MỞ RỘNG BACKEND FASTAPI CHO MUSICAI
- **Mục tiêu:** Mở rộng API backend với các router chuyên biệt cho âm nhạc tại namespace `/api/v1/music/*`.
- **Đầu vào:** Checkpoints MusicAI, `artists_clean.csv`, `music_metrics.csv`.
- **Hoạt động thực hiện:**
  - Xây dựng tầng dịch vụ: `backend/app/services/music/` (`music_service.py`, `music_model_service.py`, `music_cover_service.py`).
  - Xây dựng Pydantic schemas: `backend/app/schemas/music/` (`artist.py`, `recommendation.py`, `metrics.py`).
  - Cài đặt các API endpoints:
    - `GET /api/v1/music/artists`: Tra cứu, lọc thể loại, phân trang danh mục nghệ sĩ.
    - `GET /api/v1/music/artists/{id}`: Chi tiết nghệ sĩ, số lượt nghe, tags tiêu biểu.
    - `GET /api/v1/music/recommendations`: Sinh danh sách Top-N gợi ý nghệ sĩ theo mô hình AI chỉ định.
    - `GET /api/v1/music/models`: Danh sách các mô hình gợi ý âm nhạc và độ phức tạp.
    - `GET /api/v1/music/metrics`: Bảng đo lường 5 chỉ số khoa học của MusicAI.
    - `GET /api/v1/music/compare`: So sánh đối đầu gợi ý giữa 2 mô hình âm nhạc.
    - `GET /api/v1/music/users`: Danh sách người dùng và lịch sử nghe nhạc.
  - Tích hợp router vào `backend/app/main.py`.
- **Sản phẩm bàn giao:**
  - Toàn bộ router và dịch vụ tại `backend/app/api/v1/music/`.
  - Bộ kiểm thử tích hợp API: `tests/test_fastapi_music.py`.
- **Rào chắn CineAI:** Các router `/api/v1/movies`, `/api/v1/recommendations`, `/api/v1/metrics` hoạt động bình thường 100%.

---

### GIAI ĐOẠN M5: MỞ RỘNG GIAO DIỆN REACT 19 & SIMULATED AUDIO PLAYER
- **Mục tiêu:** Xây dựng trải nghiệm giao diện người dùng sống động cho MusicAI với Domain Switcher và trình phát nhạc chân thực.
- **Đầu vào:** Backend API MusicAI, Design System CineAI hiện hữu.
- **Hoạt động thực hiện:**
  - Cập nhật `Navbar.tsx` với thành phần `DomainSwitcher.tsx` chuyển đổi linh hoạt `[ CineAI | MusicAI ]`.
  - Phát triển 5 trang giao diện MusicAI tại `frontend/src/pages/music/`:
    1. `MusicHomePage.tsx`: Hero banner âm nhạc, bảng xếp hạng nghệ sĩ nổi bật, trending genres.
    2. `MusicDiscoverPage.tsx`: Bộ lọc nghệ sĩ theo thẻ phong cách (Rock, Pop, Jazz, Electronic, Indie...), tìm kiếm thông minh.
    3. `MusicRecommendationsPage.tsx`: Tương tác chọn người dùng, mô hình AI, số lượng $N$ để nhận gợi ý tức thì.
    4. `MusicAILabPage.tsx`: Bảng đo lường 5 chỉ số khoa học và tính năng so sánh trực tiếp 2 mô hình.
    5. `MusicProfilePage.tsx`: Hồ sơ người dùng, phân tích sở thích âm nhạc và lịch sử nghe.
  - Xây dựng linh kiện `AudioPlayerBar.tsx` (Trình phát nhạc cố định đáy trang kèm đĩa than xoay và hiệu ứng sóng âm `WaveformVisualizer.tsx`).
  - Cài đặt `musicApi.ts` và định nghĩa types `types/music.ts`.
- **Sản phẩm bàn giao:**
  - 5 trang MusicAI và các component trực quan trong `frontend/src/`.
  - Trải nghiệm chuyển đổi mượt mà không cần tải lại toàn bộ trang (SPA state).
- **Rào chắn CineAI:** Toàn bộ 5 trang phim của CineAI giữ nguyên vẹn trải nghiệm và chức năng.

---

### GIAI ĐOẠN M6: KIỂM THỬ TOÀN DIỆN HỆ THỐNG VÀ NGHIỆM THU ĐA MIỀN
- **Mục tiêu:** Kiểm toán chất lượng đầu cuối (End-to-End Audit) trên cả hai miền CineAI và MusicAI, chuẩn bị hồ sơ nghiệm thu đồ án.
- **Đầu vào:** Toàn bộ hệ thống backend, frontend, mô hình và dữ liệu của 2 miền.
- **Hoạt động thực hiện:**
  - Chạy toàn bộ test suite tổng hợp (CineAI + MusicAI).
  - Kiểm thử hiệu năng, độ trễ phản hồi API (< 20ms) và mức tiêu thụ tài nguyên máy trạm.
  - Kiểm thử trải nghiệm giao diện người dùng trên nhiều độ phân giải màn hình.
  - Tổng hợp hồ sơ tài liệu khoa học và tài liệu hướng dẫn bảo vệ đồ án.
- **Sản phẩm bàn giao:**
  - `docs/musicai/phase_m6_final_acceptance_report.md`.
  - Báo cáo tổng kết đồ án nền tảng gợi ý đa miền (Multi-Domain AI Recommendation System).

---

## 3. BẢNG DỰ TOÁN THỜI GIAN VÀ TÀI NGUYÊN HỆ THỐNG

| Giai đoạn | Thời gian dự kiến | Mức chiếm dụng RAM | Dung lượng ổ đĩa | Rủi ro chính & Biện pháp khắc phục |
|---|---|---|---|---|
| **Phase M0** | Đã hoàn thành | < 100 MB | ~1 MB (Docs) | Không có rủi ro (Chỉ sinh tài liệu kiến trúc) |
| **Phase M1** | ~1 giờ | ~300 MB | ~50 MB (Data) | Rủi ro liên kết tải Last.fm $\rightarrow$ Chuẩn bị link trực tiếp hoặc fallback |
| **Phase M2** | ~2 giờ | ~600 MB | ~30 MB (Models) | Rủi ro hội tụ iALS/NeuMF $\rightarrow$ Sử dụng early stopping và gradient clipping |
| **Phase M3** | ~1 giờ | ~400 MB | ~1 MB (Metrics) | Không có rủi ro (Thuần túy tính toán số đo) |
| **Phase M4** | ~2 giờ | ~400 MB | ~5 MB (Code) | Rủi ro xung đột route $\rightarrow$ Cô lập prefix `/api/v1/music/` |
| **Phase M5** | ~3 giờ | ~500 MB | ~10 MB (UI) | Rủi ro chồng lấn CSS $\rightarrow$ Đặt tên component và class module hóa rõ ràng |
| **Phase M6** | ~1 giờ | ~600 MB | ~2 MB (Audit) | Không có rủi ro (Nghiệm thu tổng thể) |

---

## 4. MA TRẬN RÀO CHẮN BẢO VỆ CINEAI (REGRESSION GUARDRAILS)

Trước khi đóng bất kỳ giai đoạn nào từ M1 đến M6, hệ thống tự động kiểm tra bảng kiểm tra (Checklist) gồm 5 điều kiện tiên quyết:

1. `pytest tests/test_fastapi_backend.py` $\rightarrow$ **PASS 100%**.
2. `pytest tests/test_phase4c_audit.py` $\rightarrow$ **PASS 100%**.
3. `pytest tests/test_phase5_app.py` $\rightarrow$ **PASS 100%**.
4. Toàn bộ 10 tệp checkpoint trong `models/checkpoints/*.pkl` có mã băm/dung lượng không đổi.
5. Tập tin `data/processed/train.csv` và `results/phase4c_metrics.csv` nguyên vẹn 100%.

---

## 5. KẾT LUẬN

Lộ trình M0 - M6 được thiết kế khoa học, thực tế, tối ưu hóa cho điều kiện phần cứng máy tính cá nhân và tuân thủ các chuẩn mực phát triển phần mềm tiên tiến nhất. Hệ thống sẵn sàng kích hoạt **Phase M1** ngay khi có tín hiệu bắt đầu từ người dùng.
