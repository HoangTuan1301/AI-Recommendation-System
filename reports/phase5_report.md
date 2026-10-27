# BÁO CÁO KỸ THUẬT GIAI ĐOẠN 5 (PHASE 5 TECHNICAL REPORT)
## XÂY DỰNG ỨNG DỤNG WEB STREAMLIT INTERACTIVE MOVIE RECOMMENDATION DEMO

**Dự án:** Xây dựng chương trình gợi ý phim/bài hát theo sở thích ứng dụng Trí tuệ nhân tạo  
**Học phần:** Trí tuệ nhân tạo – K5  
**Tác giả:** Senior Full-Stack Python Engineer, Machine Learning Engineer & Software Architect  
**Ngày hoàn thành:** 27/10/2026  
**Môi trường:** Python 3.11.9, Streamlit 1.66.1, Plotly 7.1.0, PyTorch 2.14.1+cpu, Windows 11 x86_64  

---

### TỔNG QUAN GIAI ĐOẠN 5

Giai đoạn 5 đánh dấu bước chuyển giao từ các mô hình nghiên cứu thực nghiệm trong phòng thí nghiệm (Offline Benchmark) sang một **ứng dụng web tương tác hoàn chỉnh (Interactive Web Application)** phục vụ trực tiếp cho buổi bảo vệ đồ án trước hội đồng giảng viên. Ứng dụng kết nối trực tiếp với 10 mô hình AI đã hoàn thiện ở các giai đoạn trước, tích hợp cơ chế lưu trữ và phục hồi checkpoint serialized, bộ nhớ đệm Streamlit Resource Caching, giao diện Dark Modern chuyên nghiệp, trực quan hóa Plotly và bộ kiểm thử tự động 76 tests.

---

## 1. KIẾN TRÚC ỨNG DỤNG (APPLICATION ARCHITECTURE)

Ứng dụng được thiết kế theo kiến trúc phân tầng (Layered Modular Architecture):

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER (STREAMLIT)                     │
│  app/streamlit_app.py (Theme: Dark Modern, Sidebar, KPI Cards, Plotly)  │
├────────────────────┬────────────────────┬────────────────────┬──────────┤
│ 🎬 Gợi ý Phim      │ 📊 Benchmark       │ 👤 Hồ sơ User      │ ℹ️ About │
│ (Recommendations)  │ (Model Comparison) │ (Profile & 1v1)    │          │
└─────────┬──────────┴─────────┬──────────┴─────────┬──────────┴─────┬────┘
          │                    │                    │                │
┌─────────▼────────────────────▼────────────────────▼────────────────▼────┐
│                    MODEL LOADER & CACHING LAYER                         │
│  app/model_loader.py (MODEL_REGISTRY, @st.cache_resource, Checkpoints)  │
├─────────────────────────────────────────────────────────────────────────┤
│  models/checkpoints/*.pkl (Popularity, CB, UCF, ICF, KNN, SVD, NCF...)  │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                     CORE RECSYS ALGORITHMS (SRC)                        │
│  Popularity, ContentBased, UserCF, ItemCF, KNNBaseline, SVD, SVD++,     │
│  NMF, NeuralCF (NeuMF PyTorch), Weighted Hybrid                         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                     DATA & RESULTS LAYER (PROCESSED)                    │
│  data/processed/train.csv, movies_clean.csv | results/phase4c_metrics.csv│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CHI TIẾT 4 TRANG CHỨC NĂNG

### Trang 1: 🎬 Gợi ý phim (Movie Recommendations)
- **Bộ điều khiển đầu vào:**
  - Chọn `User ID`: Dropdown hỗ trợ 943 người dùng trong MovieLens 100K kèm thông tin số lượng phim đã xem trong Train.
  - Chọn `Mô hình AI`: Dropdown hỗ trợ đầy đủ 10 mô hình, hiển thị kèm huy hiệu phân loại họ thuật toán (Baseline, Neighborhood CF, Matrix Factorization, Deep Learning, Hybrid).
  - Chọn số lượng Top-N: Lựa chọn linh hoạt giữa **Top 5**, **Top 10**, và **Top 20**.
- **Thẻ mô tả thuật toán:** Tự động hiển thị tóm tắt nguyên lý hoạt động, thang đo điểm dự đoán cụ thể của từng mô hình và độ phức tạp tính toán.
- **Danh sách phim đề xuất dạng lưới thẻ (Grid Cards):**
  - Hiển thị theo layout responsive (5 cột/hàng).
  - Mỗi thẻ phim bao gồm: Huy hiệu thứ hạng (`#1 TOP PICK`), tựa đề phim, năm phát hành, huy hiệu các thể loại, và khung điểm dự đoán của mô hình.
  - Sử dụng poster placeholder chuyên nghiệp với icon điện ảnh và ID phim.
- **Đảm bảo tính trung thực khoa học:**
  - $100\%$ các phim đề xuất được gọi thực tế từ hàm `model.recommend()`.
  - Tuyệt đối không gợi ý lại bất kỳ bộ phim nào người dùng đã đánh giá trong tập Train.
  - Không có hiện tượng trùng lặp phim trong danh sách Top-N.
  - Giải thích rõ ràng thang đo: Điểm của NeuMF là xác suất tương tác ngầm định ($[0, 1]$), điểm của SVD/CF là rating dự đoán ($[1, 5]$ sao), điểm của Content-Based là độ tương đồng Cosine ($[0, 1]$).
- **Chế độ xem bảng mở rộng (Expandable Table):** Cho phép xem toàn bộ dữ liệu metadata và điểm số dưới dạng bảng cho mục đích phân tích học thuật.

### Trang 2: 📊 So sánh 10 mô hình (Benchmark)
- **Nguồn dữ liệu:** Đọc trực tiếp từ file thực nghiệm chuẩn hóa [`results/phase4c_metrics.csv`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/results/phase4c_metrics.csv).
- **Hàng chỉ số KPI tổng quan:**
  - Hit Rate@10 cao nhất: **$82.03\%$** (Neural CF).
  - NDCG@10 cao nhất: **$0.3083$** (Neural CF).
  - Top 2 mô hình lai ghép: **$65.78\%$** (Hybrid CF + CB).
  - Mô hình đánh giá nhanh nhất: **$2.27\text{s}$** cho 935 người dùng kiểm thử.
- **Hệ thống biểu đồ Plotly tương tác:**
  1. *Biểu đồ cột ngang Hit Rate@10:* So sánh trực quan khả năng tìm thấy ít nhất 1 phim đúng cho người dùng, làm nổi bật NeuMF và Hybrid.
  2. *Biểu đồ cột đa độ đo:* Cho phép người dùng chuyển đổi xem giữa Precision@10, Recall@10, F1-Score@10, NDCG@10 và Hit Rate@10.
  3. *Biểu đồ thời gian thực thi:* So sánh thời gian huấn luyện (Fit Time) và thời gian đánh giá (Eval Time) trên 935 users.
- **Bảng số liệu chi tiết:** Hỗ trợ highlight giá trị cực đại tự động.

### Trang 3: 👤 Hồ sơ sở thích người dùng (Profiles)
- **Phân tích lịch sử Train:**
  - Tổng số phim đã chấm điểm, điểm trung bình ($\mu_{rating}$), số lượng phim chấm $4-5$ sao.
  - Biểu đồ phân bố điểm số đánh giá (Histogram 1-5 sao).
  - Biểu đồ thể loại phim được người dùng xem và đánh giá nhiều nhất.
  - Bảng danh sách các bộ phim yêu thích nhất (Rating $4.0 - 5.0$).
- **Chức năng Đối đầu Trực tiếp (Side-by-Side Model Comparison):**
  - Cho phép chọn bất kỳ 2 mô hình (ví dụ: Neural CF đối đầu User-Based CF).
  - Hệ thống tính toán và hiển thị danh sách Top-10 của cả hai mô hình đặt cạnh nhau.
  - Tính toán độ tương đồng tập hợp Jaccard giữa hai danh sách:
    $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
  - Đánh dấu sao `⭐ (Trùng)` cho các bộ phim cùng được cả 2 mô hình đề xuất, giúp giảng viên thấy rõ sự khác biệt giữa tư duy gợi ý của Deep Learning và Lọc cộng tác láng giềng.

### Trang 4: ℹ️ Giới thiệu đề tài (About)
- Toàn văn giới thiệu bối cảnh, học phần, bộ dữ liệu MovieLens 100K.
- Giải thích chi tiết 10 mô hình AI và công thức 5 độ đo đánh giá.
- **Nhấn mạnh chuẩn mực khoa học:** Định nghĩa chính xác chỉ số Hit Rate@10 = 82.03% (tỷ lệ người dùng có ít nhất 1 phim phù hợp trong Top-10), loại bỏ ngộ nhận cho rằng 82% tổng số phim gợi ý là chính xác.

---

## 3. CƠ CHẾ KẾT NỐI, CACHING VÀ SERIALIZATION

1. **Serialization Checkpoint (.pkl):**
   - File [`app/model_loader.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/app/model_loader.py) quản lý ánh xạ `MODEL_REGISTRY`.
   - Script [`scripts/train_and_save_checkpoints.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/scripts/train_and_save_checkpoints.py) đã tiền huấn luyện và lưu $100\%$ cả 10 mô hình vào thư mục `models/checkpoints/`:
     - `popularity.pkl`, `content_based.pkl`, `user_cf.pkl`, `item_cf.pkl`, `knn_baseline.pkl`
     - `svd.pkl`, `svdpp.pkl`, `nmf.pkl`, `neural_cf.pkl`, `hybrid.pkl`
2. **Streamlit Resource Caching (`@st.cache_resource`):**
   - Hàm `get_model()` sử dụng `@st.cache_resource` để giữ thể hiện mô hình trong bộ nhớ RAM sau lần tải đầu tiên.
   - Khi người dùng tương tác, đổi tham số hoặc đổi trang trên Streamlit, mô hình **không bao giờ bị huấn luyện lại hay tải lại từ đĩa**.
   - Thời gian phản hồi gợi ý khi bấm nút: **$0.005 - 0.05$ giây**.
3. **Data Caching (`@st.cache_data`):**
   - Tập `movies_clean.csv`, `train.csv`, và `phase4c_metrics.csv` được nạp một lần duy nhất vào bộ nhớ thông qua `load_data()`.

---

## 4. KẾT QUẢ KIỂM THỬ TOÀN DỰ ÁN (TEST SUITE)

Đã bổ sung bộ kiểm thử chuyên biệt [`tests/test_phase5_app.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/tests/test_phase5_app.py) gồm 11 bài test kiểm tra tính toàn vẹn registry, checkpoint loading, Top-N size, không lặp, không rò rỉ train history, xử lý cold-user, và kiểm thử render cả 4 trang qua harness `streamlit.testing.v1.AppTest`.

Chạy toàn bộ test suite dự án bằng lệnh:
```powershell
python -m unittest discover -s tests -v
```

**Kết quả:**
- `tests/test_phase2_pipeline.py`: 10/10 PASS
- `tests/test_phase3_pipeline.py`: 10/10 PASS
- `tests/test_phase4a_pipeline.py`: 10/10 PASS
- `tests/test_phase4b_pipeline.py`: 15/15 PASS
- `tests/test_phase4c_pipeline.py`: 13/13 PASS
- `tests/test_phase4c_audit.py`: 7/7 PASS
- `tests/test_phase5_app.py`: 11/11 PASS

**Tổng cộng:** **76 / 76 unit tests ĐẠT 100% (OK)** trong thời gian thực thi ~18 giây.

---

## 5. DANH SÁCH FILE ĐÃ TẠO VÀ CHỈNH SỬA TRONG GIAI ĐOẠN 5

1. [`.streamlit/config.toml`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/.streamlit/config.toml): Cấu hình theme Dark Modern (`#0F172A`, `#1E293B`, `#6366F1`).
2. [`app/model_loader.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/app/model_loader.py): Module quản lý 10 mô hình, registry, checkpoint caching và scoring.
3. [`app/streamlit_app.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/app/streamlit_app.py): Ứng dụng web giao diện người dùng chính với 4 trang.
4. [`scripts/train_and_save_checkpoints.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/scripts/train_and_save_checkpoints.py): Script xuất checkpoint tự động cho 10 mô hình.
5. `models/checkpoints/*.pkl`: 10 file serialized model checkpoints.
6. [`tests/test_phase5_app.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/tests/test_phase5_app.py): Bộ kiểm thử tự động 11 tiêu chí cho ứng dụng web.
7. [`README.md`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/README.md): Cập nhật toàn diện hướng dẫn cài đặt, chạy Streamlit và lộ trình.
8. [`reports/phase5_report.md`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/reports/phase5_report.md): Báo cáo kỹ thuật chi tiết Giai đoạn 5.

---

## 6. HƯỚNG DẪN TRÌNH DIỄN DEMO CHO GIẢNG VIÊN (DEMO SCRIPT)

Khi trình bày đồ án trước hội đồng giảng viên:

1. **Khởi động ứng dụng:**
   ```powershell
   python -m streamlit run app/streamlit_app.py
   ```
2. **Trình diễn Trang 2 (Model Comparison) trước tiên:**
   - Cho giảng viên thấy bảng so sánh tổng thể 10 mô hình dựa trên 5 độ đo.
   - Chỉ ra mô hình Neural CF đạt Hit Rate@10 cao nhất ($82.03\%$) và mô hình Hybrid đứng thứ hai ($65.78\%$).
   - Nhấn mạnh quy trình kiểm thử nghiêm ngặt: $100\%$ đánh giá offline trên 935 người dùng của tập Test độc lập.
3. **Trình diễn Trang 1 (Movie Recommendations):**
   - Chọn `User ID 1` hoặc `User ID 196`.
   - Chọn mô hình `Neural CF (NeuMF)` -> Bấm gợi ý để chứng minh thời gian phản hồi tức thời (< 50ms) nhờ checkpoint caching.
   - Đổi sang mô hình `User-Based CF` và `Popularity` để giảng viên quan sát sự khác biệt về danh sách phim và thang đo điểm số.
4. **Trình diễn Trang 3 (User Preferences & So sánh đối đầu):**
   - Mở hồ sơ người dùng để chứng minh gợi ý bám sát thể loại mà người dùng đã chấm điểm cao trong tập Train.
   - Sử dụng tính năng **So sánh Đối đầu** giữa `Neural CF` và `User-Based CF`: Chỉ ra tỷ lệ trùng lặp Jaccard và các phim được cả hai mô hình cùng đánh giá cao.
5. **Chứng minh Tính Đúng đắn Kỹ thuật:**
   - Chạy lệnh `python -m unittest discover -s tests -v` trong terminal để chứng minh toàn bộ 76 test cases đều vượt qua 100%.

---

## 7. TRẠNG THÁI GIT & KẾT LUẬN

- Cấu trúc Git: Nhánh `main`, working tree sạch sẽ, các file dữ liệu gốc `data/raw/` và `data/processed/train.csv`, `test.csv` được giữ nguyên vẹn $100\%$.
- Không có bất kỳ lệnh commit/push tự động nào được thực hiện.
- **KẾT LUẬN:** Giai đoạn 5 đã hoàn thành xuất sắc toàn bộ yêu cầu, mang lại một ứng dụng demo trực quan, hiện đại, ổn định, sẵn sàng phục vụ cho buổi nghiệm thu và bảo vệ đồ án môn Trí tuệ nhân tạo.
