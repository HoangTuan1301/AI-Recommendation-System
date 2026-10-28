# BẢNG SO SÁNH BENCHMARK THẨM ĐỊNH (VALIDATION BENCHMARK)
## PHASE M2: 6 MUSICAI MODELS VALIDATION COMPARISON (FULL-CATALOG TOP-10)

**Giao thức đánh giá:**
- **Tập dữ liệu:** `data/music/processed/validation.csv` (4,844 tương tác thực tế của 2,520 người dùng).
- **Tập ứng viên (Candidate Pool):** Toàn bộ danh mục 2,706 bài hát trừ đi lịch sử nghe trong tập Train (`exclude_seen=True`).
- **Xếp hạng:** Top-10 bài hát có điểm cao nhất.
- **Phương pháp bình quân:** Macro-Average trên toàn bộ 2,520 người dùng thẩm định.

---

## 1. BẢNG XẾP HẠNG THẨM ĐỊNH 6 MÔ HÌNH (SẮP XẾP THEO NDCG@10)

| Thứ hạng | Mô hình AI | Hit Rate@10 | Precision@10 | Recall@10 | NDCG@10 | MRR@10 |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **#1** | **Item-Based CF** | **0.2425** | 0.0301 | 0.1532 | **0.1038** | **0.1144** |
| **#2** | **User-Based CF** | **0.2278** | 0.0284 | 0.1393 | **0.0996** | **0.1176** |
| **#3** | **Implicit ALS** | **0.2063** | 0.0241 | 0.1273 | **0.0812** | **0.0859** |
| **#4** | **BPR-MF** | **0.1631** | 0.0200 | 0.0906 | **0.0624** | **0.0755** |
| **#5** | **Implicit Neural CF** | **0.1214** | 0.0150 | 0.0653 | **0.0438** | **0.0521** |
| **#6** | **Popularity Baseline** | **0.1028** | 0.0121 | 0.0550 | **0.0349** | **0.0405** |

---

## 2. PHÂN TÍCH CHUYÊN SÂU HIỆU NĂNG THUẬT TOÁN

1. **Hiệu năng của các mô hình nhân tố ẩn (iALS & BPR-MF):**
   - Các thuật toán Matrix Factorization khai thác triệt để hàm độ tin cậy logarit $c_{ui} = 1 + lpha \log(1 + r_{ui})$ và kỹ thuật tối ưu xếp hạng cặp (Pairwise Ranking).
   - Nhờ đó, chúng vượt trội hơn hẳn so với baseline độ phổ biến ngẫu nhiên, đưa các bài hát liên quan lên đầu bảng xếp hạng.
2. **Hiệu năng của mô hình lân cận (Item-Based CF vs User-Based CF):**
   - Item-Based CF có độ ổn định cao nhờ tập láng giềng đồng nghe (co-listening) của 2,706 bài hát đã được làm giàu qua quá trình lọc lõi (dense subgraph).
3. **Hiệu năng của mô hình học sâu NeuMF:**
   - Kết hợp phi tuyến tính giữa GMF và MLP giúp mô hình nắm bắt được các tương tác phức tạp ở cấp độ biểu diễn đặc trưng.
