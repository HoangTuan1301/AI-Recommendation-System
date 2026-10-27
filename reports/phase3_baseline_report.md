# Báo Cáo Thực Nghiệm Giai Đoạn 3: Evaluation Framework & Baseline Recommenders

**Đề tài:** Xây dựng chương trình gợi ý phim/bài hát theo sở thích ứng dụng Trí tuệ nhân tạo (AI Recommendation System)  
**Tác giả:** Senior Machine Learning Engineer & Recommender Systems Researcher  
**Dự án:** AI-Recommendation-System - Học kỳ 5  
**Tập dữ liệu:** MovieLens 100K (Train: 80,000 / Test: 20,000)  

---

## 1. Tổng Quan Kiến Trúc Evaluation Framework

Trong Giai đoạn 3, hệ thống đã thiết lập một khung đánh giá chuẩn hóa (Standardized Top-N Evaluation Framework) phục vụ cho việc đối sánh công bằng, khách quan giữa **10 mô hình gợi ý AI** trên cùng một tập dữ liệu kiểm thử độc lập (Test Set).

### Thành phần cấu trúc:
1. `src/evaluation/metrics.py`: Triển khai chính xác 5 độ đo xếp hạng Top-K với độ phức tạp tối ưu, kiểm soát hoàn toàn các điều kiện biên (chia cho 0, tập rỗng, K vượt ngưỡng).
2. `src/evaluation/evaluator.py`: Lớp điều phối `TopNEvaluator` đảm nhiệm toàn bộ quy trình sinh candidate, gọi mô hình gợi ý, lọc dữ liệu đã xem, tính toán độ đo và tổng hợp Macro-Average trên từng người dùng.
3. `src/models/base_recommender.py`: Giao diện trừu tượng chuẩn hóa (`fit`, `recommend`, `predict`) làm kim chỉ nam bắt buộc cho tất cả các mô hình triển khai trong dự án.

---

## 2. Công Thức Và Cơ Chế Tính Toán 5 Độ Đo Xếp Hạng (K = 10)

Ký hiệu $\mathcal{R}_u(K)$ là danh sách $K = 10$ phim do mô hình đề xuất cho người dùng $u$, và $\mathcal{I}_u^{test\_rel}$ là tập hợp các bộ phim thực sự yêu thích trong Test ($r_{ui} \ge 4.0$).

### 2.1. Precision@10 (Độ chính xác)
Tỷ lệ số item phù hợp xuất hiện trong top 10 gợi ý:
$$\text{Precision@}10(u) = \frac{|\mathcal{R}_u(10) \cap \mathcal{I}_u^{test\_rel}|}{10}$$

### 2.2. Recall@10 (Độ bao quát / Phủ)
Tỷ lệ số item phù hợp tìm trúng so với tổng số item người dùng yêu thích trong Test:
$$\text{Recall@}10(u) = \frac{|\mathcal{R}_u(10) \cap \mathcal{I}_u^{test\_rel}|}{|\mathcal{I}_u^{test\_rel}|}$$

### 2.3. F1@10 (Trung bình điều hòa F1)
Cân bằng giữa độ chính xác và độ bao quát:
$$\text{F1@}10(u) = \begin{cases} 2 \cdot \frac{\text{Precision@}10(u) \cdot \text{Recall@}10(u)}{\text{Precision@}10(u) + \text{Recall@}10(u)} & \text{khi } P + R > 0 \\ 0 & \text{khi } P + R = 0 \end{cases}$$

### 2.4. Hit Rate@10 (Tỷ lệ tìm trúng)
Đo lường xác suất hệ thống gợi ý trúng ít nhất một phim phù hợp trong top 10:
$$\text{Hit Rate@}10(u) = \begin{cases} 1.0 & \text{nếu } |\mathcal{R}_u(10) \cap \mathcal{I}_u^{test\_rel}| > 0 \\ 0.0 & \text{ngược lại} \end{cases}$$

### 2.5. NDCG@10 (Normalized Discounted Cumulative Gain)
Độ đo xếp hạng có tính đến vị trí xuất hiện (rank position) của item được gợi ý:
$$\text{DCG@}10(u) = \sum_{i=1}^{10} \frac{\mathbb{I}(i_k \in \mathcal{I}_u^{test\_rel})}{\log_2(i + 1)}$$
$$\text{IDCG@}10(u) = \sum_{i=1}^{\min(10, |\mathcal{I}_u^{test\_rel}|)} \frac{1}{\log_2(i + 1)}$$
$$\text{NDCG@}10(u) = \begin{cases} \frac{\text{DCG@}10(u)}{\text{IDCG@}10(u)} & \text{nếu } \text{IDCG@}10(u) > 0 \\ 0 & \text{ngược lại} \end{cases}$$

---

## 3. Không Gian Ứng Viên & Quy Chuẩn Đánh Giá (Evaluation Policy)

- **Không gian ứng viên (Candidate Universe):** Với mỗi người dùng $u$, danh sách ứng viên đề xuất được xác định là toàn bộ 1,682 phim trong danh mục trừ đi các phim người dùng đã đánh giá trong tập Train:
$$\mathcal{C}_u = \mathcal{I}_{all} \setminus \mathcal{I}_u^{train}$$
- **Quy tắc không rò rỉ (Strict Train Masking):** Cả hai mô hình đều không được phép đề xuất lại phim đã có trong lịch sử Train. Evaluator được tích hợp cơ chế fail-safe tự động lọc bỏ mọi phim trùng lặp hoặc đã xem trong Train.
- **Tập người dùng đánh giá ($\mathcal{U}_{eval}$):**
  - Trong 943 người dùng của tập Test, có chính xác **935 người dùng** có ít nhất một đánh giá tích cực ($\ge 4.0$).
  - 8 người dùng còn lại chỉ chấm điểm 1, 2, 3 sao trong Test, không có ground-truth tích cực để đo lường.
  - Theo đúng chuẩn học thuật, các độ đo Top-10 được tính trung bình Macro-Average trên chính xác **935 người dùng** này:
$$\text{Metric@}10 = \frac{1}{|\mathcal{U}_{eval}|} \sum_{u \in \mathcal{U}_{eval}} \text{Metric@}10(u) \quad (|\mathcal{U}_{eval}| = 935)$$

---

## 4. Mô Tả Hai Mô Hình Baseline

### 4.1. Popularity Recommender (Mô hình Gợi ý Phổ biến)
- **Cơ chế:** Gợi ý các bộ phim được cộng đồng người dùng đánh giá cao nhất và có lượng tương tác lớn nhất trong tập Train.
- **Công thức tính điểm:** Áp dụng công thức Bayesian Weighted Rating (chuẩn IMDb) nhằm cân bằng giữa số lượng đánh giá ($v$) và điểm trung bình ($R$):
$$\text{WR} = \left(\frac{v}{v + m}\right) R + \left(\frac{m}{v + m}\right) C$$
Trong đó:
  - $v$: Số lượt đánh giá của phim trong tập Train.
  - $m$: Ngưỡng lọc tối thiểu (lấy theo phân vị 25% của số lượng đánh giá các phim).
  - $R$: Điểm đánh giá trung bình của phim trong Train.
  - $C$: Điểm trung bình toàn cục của toàn bộ tập Train ($C \approx 3.5254$).
- **Xử lý đồng hạng (Tie-breaking):** Sắp xếp giảm dần theo điểm số `score`, nếu trùng điểm thì sắp xếp tăng dần theo `movie_id` để đảm bảo tính tất định (deterministic).

### 4.2. Content-Based Recommender (Mô hình Gợi ý Dựa trên Nội dung)
- **Cơ chế:** Phân tích nội dung phim và xây dựng hồ sơ sở thích cá nhân hóa cho từng người dùng.
- **Biểu diễn nội dung phim (Item Representations):**
  - Trích xuất đặc trưng từ 19 thể loại kết hợp với tên phim (title tokens) thông qua thuật toán **TF-IDF Vectorizer** (tạo không gian đặc trưng gồm 2,350 chiều).
  - Chuẩn hóa vector $L_2$ để tính độ tương đồng Cosine trực tiếp qua tích vô hướng: $\cos(\mathbf{p}, \mathbf{v}) = \mathbf{p} \cdot \mathbf{v}$.
- **Xây dựng hồ sơ người dùng (User Preference Profile):**
  - Áp dụng phương pháp Mean-Centered Weighting từ lịch sử Train của người dùng:
$$w_{ui} = r_{ui} - \bar{r}_u$$
  - Các phim người dùng chấm điểm cao hơn mức trung bình cá nhân mang trọng số dương (thể hiện thể loại yêu thích), các phim chấm thấp hơn trung bình mang trọng số âm (đẩy xa các thể loại không thích).
  - Hồ sơ sở thích người dùng được tính toán siêu tốc thông qua phép nhân ma trận thưa - dày:
$$\mathbf{P} = \mathbf{W}_{sparse} \cdot \mathbf{V}_{tfidf}$$
  - Sau đó chuẩn hóa $L_2$: $\mathbf{p}_u \leftarrow \mathbf{p}_u / \|\mathbf{p}_u\|_2$.

---

## 5. Kết Quả Thực Nghiệm Đối Soát (Benchmark Results)

Kết quả thực nghiệm trên tập Test độc lập (lưu tại `results/phase3_baseline_metrics.csv`):

| Mô hình (Model) | Precision@10 | Recall@10 | F1@10 | NDCG@10 | Hit Rate@10 | Số Users Đánh Giá | Thời Gian Thực Thi (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity_Baseline** | **0.0653** | **0.0497** | **0.0473** | **0.0764** | **0.3914** | **935** | **0.82 s** |
| **ContentBased_Baseline** | **0.0236** | **0.0199** | **0.0174** | **0.0296** | **0.1872** | **935** | **5.74 s** |

---

## 6. Nhận Xét Chuyên Sâu Kết Quả Thực Nghiệm

1. **Hiệu năng của Popularity Baseline:**
   - Popularity Recommender đạt **Hit Rate@10 lên tới 39.14%** và NDCG@10 đạt **0.0764**, vượt trội so với Content-Based Baseline trong đánh giá ngoại tuyến.
   - *Nguyên nhân kỹ thuật:* Dữ liệu MovieLens có hiện tượng phân bố đuôi dài rất rõ nét (Long-Tail Distribution), người dùng thường có xu hướng xem và đánh giá các bộ phim bom tấn nổi tiếng (*Star Wars*, *Fargo*, *Schindler's List*). Vì vậy, các bộ phim có chất lượng cao và phổ biến rộng rãi sẽ có xác suất rơi trúng vào tập Test của nhiều người dùng khác nhau.
2. **Đặc tính của Content-Based Baseline:**
   - Content-Based Recommender đạt **Hit Rate@10 là 18.72%** và Precision@10 là **2.36%**.
   - *Nguyên nhân kỹ thuật:* 
     - Mô hình Content-Based gợi ý dựa trên sự tương đồng thể loại và từ khóa. Khi một người dùng thích thể loại *Sci-Fi* hoặc *Comedy*, mô hình sẽ gợi ý các phim Sci-Fi/Comedy ít phổ biến hơn nhưng rất giống về mặt nội dung.
     - Trong đánh giá Offline, nếu người dùng chưa từng xem bộ phim ngách đó trong thực tế (ô chưa quan sát trong Test), tương tác đó sẽ bị tính là một lần đoán trượt (False Positive).
     - Tuy nhiên, Content-Based có giá trị vô cùng to lớn trong việc **giải quyết bài toán Cold-Start cho item mới** (những phim chưa có ai đánh giá trong Train vẫn có thể được gợi ý nhờ metadata).
3. **Hiệu năng tính toán (Runtime Efficiency):**
   - Cả hai mô hình đều có tốc độ dự đoán cực nhanh (Popularity hoàn thành 935 users trong 0.82s, Content-Based trong 5.74s nhờ vector hóa ma trận thưa).

---

## 7. Những Hạn Chế Của Đánh Giá Offline Với Dữ Liệu Rating

1. **Thiên lệch dữ liệu quan sát (Missing-Not-At-Random - MNAR):** Dữ liệu MovieLens không phải là ngẫu nhiên hoàn toàn. Một bộ phim chưa có rating trong Test không có nghĩa là người dùng ghét phim đó. Do đó các chỉ số Precision@10 và Recall@10 trên tập Test mang tính chất ước lượng chặn dưới (lower-bound estimate).
2. **Thiên lệch độ phổ biến (Popularity Bias):** Mô hình phổ biến luôn chiếm ưu thế trong các chỉ số thống kê tổng thể, nhưng thiếu tính cá nhân hóa sâu sắc và không đem lại cảm giác bất ngờ, thú vị (Serendipity / Novelty) cho người dùng.

---

## 8. Các Vấn Đề Cần Chuẩn Bị Cho Giai Đoạn 4 (Phase 4)

Trong Giai đoạn 4, chúng ta sẽ bắt tay vào triển khai các mô hình nâng cao:
1. **Lọc cộng tác dựa trên độ tương đồng (Collaborative Filtering):**
   - User-Based Collaborative Filtering (User-KNN).
   - Item-Based Collaborative Filtering (Item-KNN).
   - KNN with Baselines (KNNBaseline) để loại bỏ độ lệch bias của người dùng và sản phẩm.
2. **Phân rã ma trận (Matrix Factorization):**
   - Singular Value Decomposition (SVD).
   - SVD++ (tích hợp implicit feedback).
   - Non-Negative Matrix Factorization (NMF).
3. **Mô hình Lai ghép & Học sâu (Deep Learning & Hybrid):**
   - Neural Collaborative Filtering (NCF).
   - Hybrid Recommender (kết hợp Content-Based và Collaborative Filtering).
4. **Chuẩn hóa API:** Tất cả 8 mô hình tiếp theo đều sẽ kế thừa trực tiếp từ `BaseRecommender` và được đánh giá trên cùng một thể hiện `TopNEvaluator` đã xây dựng trong Giai đoạn 3, đảm bảo tính nhất quán khoa học tuyệt đối.
