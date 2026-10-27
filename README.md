# AI-Recommendation-System
Hệ thống gợi ý phim / bài hát theo sở thích ứng dụng Trí tuệ nhân tạo (AI Recommendation System).

Dự án Bài tập lớn môn **Trí tuệ nhân tạo (AI)** - Học kỳ 5.

---

## 1. Giới thiệu đề tài

Hệ thống khuyến nghị (Recommendation System) là một phân nhánh quan trọng của Trí tuệ nhân tạo và Học máy, đóng vai trò cá nhân hóa trải nghiệm người dùng trên các nền tảng nội dung số (Netflix, Spotify, YouTube,...). Dự án này nghiên cứu, triển khai thực nghiệm và so sánh các phương pháp gợi ý khác nhau từ truyền thống đến nâng cao, ứng dụng cho bài toán gợi ý phim (MovieLens) và bài hát (Spotify).

---

## 2. Mục tiêu dự án

- **Nghiên cứu & Thực nghiệm**: Cài đặt và đánh giá toàn diện **10 mô hình/phương pháp gợi ý** (từ Heuristic, Collaborative Filtering, Content-Based đến Matrix Factorization và Deep Learning).
- **So sánh đa chiều**: Đánh giá hiệu năng của các mô hình dựa trên **5 độ đo chuẩn mực** trong hệ thống gợi ý.
- **Ứng dụng thực tế**: Xây dựng ứng dụng web demo trực quan (Streamlit) cho phép người dùng tương tác, nhận gợi ý theo thời gian thực và so sánh hiệu quả giữa các thuật toán.

---

## 3. Bộ dữ liệu (Datasets)

1. **MovieLens 100K** *(Phạm vi ưu tiên giai đoạn đầu)*:
   - Nguồn: [GroupLens Research](https://grouplens.org/datasets/movielens/100k/)
   - Quy mô: 100,000 lượt đánh giá từ 943 người dùng cho 1,682 bộ phim.
   - Đặc trưng: Đánh giá xếp hạng (1-5 sao), thông tin người dùng (tuổi, giới tính, nghề nghiệp), thông tin phim (thể loại, ngày phát hành).
2. **Spotify Tracks Dataset** *(Mở rộng sau khi thống nhất phạm vi)*:
   - Nguồn: [Kaggle Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/)
   - Đặc trưng âm học: Danceability, Energy, Key, Loudness, Acousticness, Valence, Tempo,...

---

## 4. Danh mục 10 Mô hình & 5 Độ đo đánh giá

### 10 Mô hình / Phương pháp gợi ý:
1. **Popularity-based Recommender** (Baseline - gợi ý theo độ phổ biến/điểm trung bình cao).
2. **Item-Based Collaborative Filtering** (Lọc cộng tác dựa trên độ tương đồng giữa các item).
3. **User-Based Collaborative Filtering** (Lọc cộng tác dựa trên độ tương đồng hành vi giữa người dùng).
4. **Content-Based Filtering (TF-IDF & Cosine Similarity)** (Gợi ý dựa trên nội dung, thể loại và metadata).
5. **Singular Value Decomposition (SVD)** (Phân rã ma trận Matrix Factorization kinh điển).
6. **SVD++** (Mở rộng từ SVD tích hợp thông tin phản hồi ẩn - implicit feedback).
7. **Non-negative Matrix Factorization (NMF)** (Phân rã ma trận không âm).
8. **Neural Collaborative Filtering (NCF)** (Mạng nơ-ron kết hợp GMF và MLP).
9. **K-Nearest Neighbors with Baseline (KNNBaseline)** (KNN kết hợp điều chỉnh độ lệch bias của user & item).
10. **Hybrid Recommender** (Kết hợp Content-Based và Collaborative Filtering / Weighted Hybrid).

### 5 Độ đo đánh giá (Evaluation Metrics):
1. **RMSE (Root Mean Squared Error)**: Đánh giá sai số dự đoán điểm số xếp hạng.
2. **MAE (Mean Absolute Error)**: Đánh giá sai số tuyệt đối trung bình.
3. **Precision@K**: Tỷ lệ các item được gợi ý trong top-K thực sự phù hợp với sở thích người dùng.
4. **Recall@K**: Tỷ lệ các item người dùng yêu thích được hệ thống gợi ý thành công trong top-K.
5. **NDCG@K (Normalized Discounted Cumulative Gain)**: Đánh giá chất lượng và thứ tự ưu tiên của danh sách xếp hạng top-K.

---

## 5. Công nghệ & Thư viện sử dụng

- **Ngôn ngữ**: Python (khuyến nghị Python 3.11).
- **Xử lý & Phân tích dữ liệu**: `numpy`, `pandas`, `scipy`.
- **Học máy & Khai phá dữ liệu**: `scikit-learn`.
- **Trực quan hóa dữ liệu**: `matplotlib`, `seaborn`, `plotly`.
- **Ứng dụng giao diện trực quan**: `streamlit`.
- **Môi trường thử nghiệm**: `jupyter notebook`.

---

## 6. Cấu trúc thư mục dự án

```text
AI-Recommendation-System/
├── app/                  # Mã nguồn ứng dụng demo giao diện người dùng (Streamlit)
│   └── .gitkeep
├── data/
│   ├── raw/              # Dữ liệu gốc tải về (MovieLens 100K, Spotify)
│   └── processed/        # Dữ liệu đã qua làm sạch, chuẩn hóa và trích xuất đặc trưng
├── notebooks/            # Jupyter notebooks phục vụ EDA và thử nghiệm thuật toán
├── reports/              # Báo cáo kết quả, biểu đồ, slide thuyết trình
├── results/              # Kết quả đánh giá mô hình (bảng số liệu, ma trận đối sánh)
├── src/                  # Mã nguồn chính của dự án
│   ├── evaluation/       # Module tính toán 5 độ đo đánh giá (RMSE, MAE, Precision@K,...)
│   └── models/           # Cài đặt 10 mô hình gợi ý
├── tests/                # Unit tests kiểm thử mã nguồn và pipeline
├── .gitignore            # Cấu hình bỏ qua các file tạm, dữ liệu lớn, .venv
├── README.md             # Tài liệu tổng quan dự án
└── requirements.txt      # Danh sách thư viện và phiên bản phụ thuộc
```

---

## 7. Hướng dẫn cài đặt & Thiết lập môi trường

### Bước 1: Khởi tạo môi trường ảo
```powershell
# Sử dụng Python 3.11
python -m venv .venv

# Kích hoạt môi trường ảo (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

### Bước 2: Nâng cấp pip và cài đặt thư viện
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 3: Chạy ứng dụng Streamlit (khi hoàn thiện)
```powershell
streamlit run app/main.py
```

---

## 8. Lộ trình phát triển (Roadmap)

- [x] **Giai đoạn 1**: Khởi tạo cấu trúc dự án, cấu hình môi trường, tài liệu và Git setup.
- [ ] **Giai đoạn 2**: Thu thập và tiền xử lý dữ liệu MovieLens 100K (EDA, Data Cleaning, Train/Test Split).
- [ ] **Giai đoạn 3**: Xây dựng module đánh giá chuẩn (5 độ đo: RMSE, MAE, Precision@K, Recall@K, NDCG@K).
- [ ] **Giai đoạn 4**: Cài đặt và huấn luyện 10 mô hình gợi ý.
- [ ] **Giai đoạn 5**: So sánh, tổng hợp kết quả thực nghiệm và trực quan hóa so sánh.
- [ ] **Giai đoạn 6**: Xây dựng giao diện ứng dụng demo Streamlit.
- [ ] **Giai đoạn 7**: Mở rộng dataset Spotify (nếu được phê duyệt) và hoàn thiện báo cáo môn học.
