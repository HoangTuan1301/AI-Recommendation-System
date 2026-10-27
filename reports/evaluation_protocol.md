# Giao Thức Đánh Giá Hệ Thống Gợi Ý (Evaluation Protocol)

**Dự án:** Hệ thống gợi ý phim/bài hát theo sở thích ứng dụng Trí tuệ nhân tạo (AI Recommendation System)  
**Tác giả:** Nhóm thực hiện Bài tập lớn AI - Học kỳ 5  
**Phiên bản:** 1.0 (Chuẩn hóa phục vụ Giai đoạn 3 & 4)  

---

## 1. Mục Tiêu Đánh Giá Hệ Thống Gợi Ý Top-N

Trong các ứng dụng thực tế (Netflix, Spotify, YouTube), mục tiêu chính của hệ thống gợi ý không chỉ đơn thuần là dự đoán chính xác điểm số tuyệt đối mà người dùng sẽ chấm (Rating Prediction), mà quan trọng hơn là **xếp hạng và chọn lọc ra một danh sách ngắn gồm $N$ sản phẩm tốt nhất** (Top-$N$ Recommendation) mang lại sự hài lòng cao nhất cho từng cá nhân.

Giao thức này chuẩn hóa phương pháp đánh giá ngoại tuyến (Offline Evaluation Protocol) để đảm bảo tính **công bằng, khách quan, có thể tái lập (reproducible) và nhất quán** khi so sánh **10 mô hình gợi ý** trên cùng một tập dữ liệu thử nghiệm (Test Set).

---

## 2. Xác Định Tương Tác Phù Hợp (Relevance Definition)

Trong tập dữ liệu MovieLens 100K, điểm số dao động từ 1 đến 5 sao:
- Điểm 1, 2, 3 sao: Người dùng đánh giá mức độ trung bình hoặc không hài lòng.
- Điểm 4 và 5 sao: Người dùng thực sự yêu thích và hài lòng với bộ phim.

Do đó, tương tác giữa người dùng $u$ và bộ phim $i$ trong tập dữ liệu kiểm thử $Test$ được định nghĩa là **Phù hợp (Relevant)** theo quy tắc nhị phân:

$$\text{is\_relevant}(u, i) = \begin{cases} 1 & \text{nếu } r_{ui} \ge 4.0 \\ 0 & \text{nếu } r_{ui} < 4.0 \end{cases}$$

Tập các phim thực sự phù hợp của người dùng $u$ trong tập Test được ký hiệu là:
$$\mathcal{I}_u^{test\_rel} = \{i \in \mathcal{I}_u^{test} \mid r_{ui} \ge 4.0\}$$

---

## 3. Không Gian Ứng Viên & Quy Tắc Tạo Danh Sách Top-N

### 3.1. Quy tắc loại trừ lịch sử đã xem (Train History Masking)
Người dùng thường không muốn hệ thống gợi ý lại những bộ phim họ đã từng xem và đánh giá trong quá khứ. Do đó:
- Danh sách ứng viên đề xuất cho người dùng $u$ ($\mathcal{C}_u$) phải **loại trừ toàn bộ các bộ phim đã xuất hiện trong tập Train** của người dùng đó:
$$\mathcal{C}_u = \mathcal{I}_{all} \setminus \mathcal{I}_u^{train}$$

### 3.2. Phương thức xếp hạng ứng viên (Full-Ranking vs Sampling)
Để đảm bảo độ chính xác học thuật cao nhất và tránh hiện tượng thiên lệch do lấy mẫu ngẫu nhiên (sampling bias), hệ thống áp dụng cơ chế **Full Candidate Ranking**:
- Mô hình chấm điểm dự đoán $\hat{r}_{ui}$ (hoặc điểm số ưu tiên $\hat{s}_{ui}$) cho toàn bộ các item $i \in \mathcal{C}_u$.
- Sắp xếp các item theo thứ tự điểm dự đoán giảm dần.
- Lấy ra $N = 10$ item có điểm cao nhất tạo thành danh sách khuyến nghị:
$$\mathcal{R}_u(N) = [i_1, i_2, \dots, i_N]$$

---

## 4. Định Nghĩa 5 Độ Đo Đánh Giá Top-N (Tại K = 10)

Ký hiệu $\mathcal{R}_u(K)$ là danh sách $K$ item được mô hình gợi ý cho user $u$, và $\mathcal{I}_u^{test\_rel}$ là tập hợp các item thực sự phù hợp trong test của user $u$.

### 4.1. Precision@K (Độ chính xác)
Tỷ lệ số item phù hợp xuất hiện trong danh sách $K$ gợi ý so với tổng số $K$ item được gợi ý:
$$\text{Precision@}K(u) = \frac{|\mathcal{R}_u(K) \cap \mathcal{I}_u^{test\_rel}|}{K}$$
- **Ý nghĩa:** Trả lời câu hỏi: *"Trong $K$ phim gợi ý cho người dùng, có bao nhiêu phần trăm phim thực sự đúng sở thích?"*

### 4.2. Recall@K (Độ phủ / Độ bao quát)
Tỷ lệ số item phù hợp được tìm thấy trong danh sách $K$ gợi ý so với tổng số item phù hợp có trong ground-truth test của user:
$$\text{Recall@}K(u) = \frac{|\mathcal{R}_u(K) \cap \mathcal{I}_u^{test\_rel}|}{|\mathcal{I}_u^{test\_rel}|}$$
- **Ý nghĩa:** Trả lời câu hỏi: *"Hệ thống đã bao quát được bao nhiêu phần trăm số phim yêu thích của người dùng trong top $K$?"*

### 4.3. F1-Score@K (Độ đo điều hòa F1)
Trung bình điều hòa giữa Precision@K và Recall@K nhằm cân bằng giữa độ chính xác và độ bao quát:
$$\text{F1@}K(u) = \begin{cases} 2 \cdot \frac{\text{Precision@}K(u) \cdot \text{Recall@}K(u)}{\text{Precision@}K(u) + \text{Recall@}K(u)} & \text{nếu } \text{Precision@}K + \text{Recall@}K > 0 \\ 0 & \text{ngược lại} \end{cases}$$

### 4.4. Hit Rate@K (Tỷ lệ tìm trúng)
Độ đo nhị phân thể hiện hệ thống có gợi ý trúng ít nhất một phim phù hợp trong top $K$ hay không:
$$\text{HR@}K(u) = \begin{cases} 1 & \text{nếu } |\mathcal{R}_u(K) \cap \mathcal{I}_u^{test\_rel}| > 0 \\ 0 & \text{ngược lại} \end{cases}$$
- **Ý nghĩa:** Rất quan trọng với trải nghiệm người dùng thực tế; chỉ cần 1 item trúng trong top $K$ là người dùng đã có thể quyết định click xem phim.

### 4.5. NDCG@K (Normalized Discounted Cumulative Gain)
Độ đo xếp hạng có tính đến **vị trí xuất hiện** của item. Một phim phù hợp xuất hiện ở vị trí đầu tiên (rank 1) sẽ có giá trị lớn hơn nhiều so với khi nằm ở cuối danh sách (rank 10).

- **Discounted Cumulative Gain (DCG@K):**
$$\text{DCG@}K(u) = \sum_{k=1}^K \frac{2^{\text{rel}(i_k)} - 1}{\log_2(k + 1)}$$
Trong đó $\text{rel}(i_k) = 1$ nếu $i_k \in \mathcal{I}_u^{test\_rel}$, ngược lại $\text{rel}(i_k) = 0$.

- **Ideal Discounted Cumulative Gain (IDCG@K):**
Giá trị DCG lý tưởng nhất khi toàn bộ các item phù hợp được sắp xếp hoàn hảo lên các vị trí đầu tiên:
$$\text{IDCG@}K(u) = \sum_{k=1}^{\min(K, |\mathcal{I}_u^{test\_rel}|)} \frac{2^1 - 1}{\log_2(k + 1)}$$

- **Normalized DCG (NDCG@K):**
$$\text{NDCG@}K(u) = \begin{cases} \frac{\text{DCG@}K(u)}{\text{IDCG@}K(u)} & \text{nếu } \text{IDCG@}K(u) > 0 \\ 0 & \text{ngược lại} \end{cases}$$

---

### 4.6. Các độ đo đánh giá sai số dự đoán điểm (Rating Prediction Metrics)
Bên cạnh bài toán Top-N Ranking, đối với các mô hình dự đoán trực tiếp điểm rating $\hat{r}_{ui}$ (như SVD, SVD++, NMF, Baseline Estimates), hệ thống đo lường sai số trên toàn bộ tập Test ($N_{test} = 20,000$ tương tác):

1. **Root Mean Squared Error (RMSE):**
$$\text{RMSE} = \sqrt{\frac{1}{|Test|} \sum_{(u, i) \in Test} (r_{ui} - \hat{r}_{ui})^2}$$
Độ đo nhạy cảm với các sai số lớn, phạt nặng các dự đoán lệch xa thực tế.

2. **Mean Absolute Error (MAE):**
$$\text{MAE} = \frac{1}{|Test|} \sum_{(u, i) \in Test} |r_{ui} - \hat{r}_{ui}|$$
Đo lường sai số tuyệt đối trung bình, dễ diễn giải trực quan trên thang điểm 1-5 sao.

---

## 5. Quy Tắc Tính Trung Bình Và Xử Lý Biên (Boundary Conditions)

### 5.1. Xử lý người dùng không có item phù hợp trong Test ($|\mathcal{I}_u^{test\_rel}| = 0$)
- Nếu một người dùng trong tập Test chỉ toàn đánh giá 1, 2, 3 sao (không có phim nào $\ge 4$ sao), người dùng này không có ground-truth tích cực để đo lường việc tìm trúng.
- **Thực tế trên MovieLens 100K:** Trong tập Test gồm 943 người dùng, có chính xác **8 người dùng** (User IDs: `[273, 302, 418, 544, 685, 797, 824, 832]`) không có bất kỳ đánh giá nào $\ge 4.0$.
- **Quy tắc chuẩn hóa:** 
  - Đối với độ đo dự đoán điểm số (**RMSE, MAE**): Đánh giá trên toàn bộ **943 người dùng** (20,000 tương tác kiểm thử).
  - Đối với các độ đo Top-N (**Precision@10, Recall@10, F1@10, NDCG@10, Hit Rate@10**): Chỉ tính trung bình trên tập hợp **935 người dùng** có ít nhất một item phù hợp trong Test:
$$\mathcal{U}_{eval} = \{u \in \mathcal{U}_{test} \mid |\mathcal{I}_u^{test\_rel}| \ge 1\} \quad (|\mathcal{U}_{eval}| = 935)$$
Quy tắc này giúp loại trừ triệt để lỗi chia cho 0 (`ZeroDivisionError`) khi tính Recall@K và phản ánh chính xác năng lực tìm trúng của mô hình.

### 5.2. Tính toán độ đo tổng thể (Macro-Averaging over Users)
Độ đo trung bình cuối cùng của hệ thống là trung bình cộng số học của từng người dùng (User-Level Macro Average):
$$\text{Metric@}K = \frac{1}{|\mathcal{U}_{eval}|} \sum_{u \in \mathcal{U}_{eval}} \text{Metric@}K(u)$$
Phương pháp Macro-Average đảm bảo mỗi người dùng có trọng số đóng góp như nhau, tránh việc các "heavy users" (người đánh giá quá nhiều) chi phối toàn bộ kết quả của mô hình.

---

## 6. Những Hạn Chế Của Đánh Giá Offline Với Dữ Liệu Rating

Khi đánh giá mô hình gợi ý bằng phương pháp Offline trên tập MovieLens, cần ghi nhận các hạn chế kỹ thuật sau:

1. **Thiên lệch dữ liệu quan sát (Missing-Not-At-Random - MNAR):** Người dùng có xu hướng tự chọn xem những phim họ nghĩ mình sẽ thích trước khi đánh giá. Những ô trống (unobserved) trong ma trận không đồng nghĩa với việc người dùng ghét bộ phim đó, mà có thể đơn giản là họ chưa từng biết đến bộ phim.
2. **Không đo lường được sự mới lạ và bất ngờ (Serendipity & Novelty):** Các mô hình gợi ý theo độ phổ biến cao dễ đạt điểm Recall@K tốt trong đánh giá offline vì người dùng thường xem phim bom tấn, nhưng không mang lại giá trị khám phá mới cho người dùng.
3. **Thiếu phản hồi tức thời (Feedback Loop Absence):** Đánh giá offline dựa trên snapshot lịch sử tĩnh, không thể đo lường hành vi người dùng thay đổi theo thời gian thực như trong thử nghiệm Online A/B Testing.
4. **Hiệu ứng vị trí (Position Bias):** Không xét đến cách giao diện hiển thị ảnh hưởng đến quyết định click của người dùng.
