# CHIẾN LƯỢC MÔ HÌNH HÓA VÀ THUẬT TOÁN GỢI Ý ÂM NHẠC
## PHASE M0: IMPLICIT FEEDBACK MODEL STRATEGY & EVALUATION PROTOCOL

**Dự án:** Multi-Domain AI Recommendation System (MusicAI)  
**Tài liệu:** Khung lý thuyết, Chiến lược mô hình hóa và Giao thức đánh giá Top-N  
**Trạng thái:** HOÀN THIỆN & ĐÃ ĐƯỢC CHỨNG MINH LÝ THUYẾT  
**Tác giả:** Senior AI/ML Engineer & Recommender Systems Researcher  
**Ngày:** 28/10/2026  

---

## 1. BẢN CHẤT CỦA BÀI TOÁN GỢI Ý PHẢN HỒI NGẦM ĐỊNH (IMPLICIT FEEDBACK)

Khác biệt cốt lõi lớn nhất giữa bài toán gợi ý phim (**CineAI**) và bài toán gợi ý âm nhạc (**MusicAI**) nằm ở bản chất của ma trận tương tác:

```
+------------------------------------+------------------------------------+
| CINEAI (EXPLICIT FEEDBACK)         | MUSICAI (IMPLICIT FEEDBACK)        |
+------------------------------------+------------------------------------+
| - Ma trận R_{ui} in {1, 2, 3, 4, 5}| - Ma trận C_{ui} >= 0 (Play count) |
| - Có tín hiệu tiêu cực rõ ràng    | - Không có tín hiệu tiêu cực thuần |
|   (1-2 sao = Người dùng ghét phim) | - Mục chưa nghe != Người dùng ghét |
| - Tập dữ liệu phân lớp 2 chiều     | - Bài toán One-Class CF            |
| - Tối ưu hóa sai số RMSE / MAE     | - Tối ưu hóa thứ hạng (Top-N Rank) |
+------------------------------------+------------------------------------+
```

### 1.1. Khung toán học Hu - Koren - Volinsky (ICDM 2008)

Để mô hình hóa hành vi nghe nhạc, ta ánh xạ số lượt nghe thô $r_{ui} \ge 0$ thành hai biến đại diện:
1. **Sở thích nhị phân (Binary Preference $p_{ui}$):**
   $$p_{ui} = \begin{cases} 1 & \text{nếu } r_{ui} > 0 \\ 0 & \text{nếu } r_{ui} = 0 \end{cases}$$
2. **Độ tin cậy ngầm định (Confidence Level $c_{ui}$):**
   Số lượt nghe càng lớn thì mức độ tin cậy về sở thích của người dùng càng cao:
   $$c_{ui} = 1 + \alpha \cdot \log(1 + r_{ui})$$
   (với $\alpha > 0$ là siêu tham số kiểm soát độ dốc của hàm tin cậy, thông thường $\alpha \in [10, 40]$).

### 1.2. Vấn đề Lấy mẫu Âm (Negative Sampling Theory)

Vì người dùng chỉ tương tác với một phần rất nhỏ danh mục (dưới 0.5%), phần lớn các mục còn lại là **chưa quan sát (unobserved)**. Nếu coi toàn bộ các mục chưa quan sát là số 0 (tiêu cực), mô hình sẽ bị thiên lệch nghiêm trọng (bias).
Do đó, hệ thống MusicAI sẽ áp dụng 2 chiến lược lấy mẫu âm:
- **Lấy mẫu Ngẫu nhiên Đồng đều (Uniform Negative Sampling):** Chọn ngẫu nhiên $N_{neg}$ mục từ danh mục chưa nghe của người dùng.
- **Lấy mẫu theo Độ phổ biến (Popularity-Biased Negative Sampling):** Lấy mẫu mục âm tỷ lệ thuận với tần suất xuất hiện toàn cục $P(j) \propto f_j^{0.75}$. Cơ chế này buộc mô hình phải học cách phân biệt giữa nghệ sĩ thực sự được yêu thích và nghệ sĩ chỉ đơn thuần nổi tiếng do trào lưu.

---

## 2. DANH MỤC CÁC MÔ HÌNH DỰ KIẾN TRIỂN KHAI CHO MUSICAI (PHASE M2)

Hệ thống đề xuất 6 mô hình toán học và học sâu đại diện cho đầy đủ các trường phái thuật toán hàng đầu trong lĩnh vực Implicit Recommendation:

```
                                [ MUSICAI MODEL SUITE ]
                                           │
         ┌────────────────────────┬────────┴────────┬────────────────────────┐
         ▼                        ▼                 ▼                        ▼
    [ BASELINE ]          [ NEIGHBORHOOD ]   [ FACTORIZATION ]         [ DEEP LEARNING ]
  - Global Popularity   - Item-Based CF     - Implicit ALS (iALS)    - Implicit NeuMF
                        - User-Based CF     - BPR-MF (Pairwise)      - Tag-Based Hybrid
```

### 2.1. Mô hình 1: Global Popularity Baseline
- **Ý tưởng:** Xếp hạng các nghệ sĩ/bài hát dựa trên tổng số lượt nghe tích lũy hoặc số lượng thính giả độc nhất (Unique Listeners):
  $$\text{Score}(i) = \sum_{u \in U} \log(1 + r_{ui})$$
- **Mục đích:** Đóng vai trò là mốc chuẩn (Lower-bound Benchmark) không cá nhân hóa.

### 2.2. Mô hình 2: Item-Based Implicit Collaborative Filtering
- **Ý tưởng:** Tính độ tương đồng giữa hai bài hát/nghệ sĩ dựa trên véc-tơ thói quen nghe của cộng đồng.
- **Độ đo tương đồng:** Sử dụng chỉ số Cosine chuẩn hóa hoặc Jaccard Similarity trên tập người nghe chung:
  $$\text{Sim}_{\text{Jaccard}}(i, j) = \frac{|U_i \cap U_j|}{|U_i \cup U_j|}$$
- **Dự đoán:**
  $$\hat{s}(u, i) = \sum_{j \in \mathcal{H}_u \cap \mathcal{N}_k(i)} \text{Sim}(i, j) \cdot c_{uj}$$

### 2.3. Mô hình 3: User-Based Implicit Collaborative Filtering
- **Ý tưởng:** Tìm $K$ người dùng có gu âm nhạc tương đồng nhất dựa trên tập hợp bài hát đã nghe, sau đó tổng hợp các bài hát họ yêu thích mà người dùng mục tiêu chưa từng trải nghiệm.

### 2.4. Mô hình 4: Implicit Alternating Least Squares (iALS)
- **Công trình gốc:** Hu, Koren, Volinsky (2008) - bài báo nền tảng của mọi hệ thống gợi ý ngầm định hiện đại.
- **Hàm mất mát:**
  $$\mathcal{L}_{\text{iALS}} = \sum_{u \in U} \sum_{i \in I} c_{ui} (p_{ui} - \mathbf{x}_u^T \mathbf{y}_i)^2 + \lambda \left( \sum_u \|\mathbf{x}_u\|_2^2 + \sum_i \|\mathbf{y}_i\|_2^2 \right)$$
- **Nghiệm giải tích (Closed-form Update):**
  Nhờ thủ thuật tính toán trước ma trận tích lũy $Y^T Y = \sum_i \mathbf{y}_i \mathbf{y}_i^T$ (độ phức tạp giảm từ $O(|I| k^3)$ xuống $O(k^3)$), mỗi bước cập nhật nhân tố người dùng được giải nhanh chóng:
  $$\mathbf{x}_u = \left( Y^T Y + Y^T (C^u - I) Y + \lambda I_k \right)^{-1} Y^T C^u \mathbf{p}_u$$

### 2.5. Mô hình 5: Bayesian Personalized Ranking Matrix Factorization (BPR-MF)
- **Công trình gốc:** Rendle, Freudenthaler, Gantner, Schmidt-Thieme (UAI 2009).
- **Nguyên lý Pairwise Ranking:** Thay vì dự đoán điểm số tuyệt đối, BPR tối ưu hóa xác suất người dùng $u$ thích mục đã nghe $i$ hơn mục chưa quan sát $j$ ($i >_u j$):
  $$p(i >_u j \mid \Theta) = \sigma(\hat{x}_{ui} - \hat{x}_{uj}) = \frac{1}{1 + e^{-(\hat{x}_{ui} - \hat{x}_{uj})}}$$
- **Hàm mục tiêu BPR-Opt:**
  $$\max_{\Theta} \sum_{(u, i, j) \in D_S} \ln \sigma(\mathbf{x}_u^T \mathbf{y}_i - \mathbf{x}_u^T \mathbf{y}_j) - \frac{\lambda_\Theta}{2} \|\Theta\|_2^2$$
- Huấn luyện hiệu quả thông qua giải thuật Stochastic Gradient Descent (SGD) với cặp lấy mẫu bộ ba $(u, i, j)$.

### 2.6. Mô hình 6: Neural Collaborative Filtering for Implicit Data (Implicit NeuMF)
- **Cấu trúc mạng:** Kết hợp nhánh nhân tố ma trận tổng quát (**GMF**) và mạng nơ-ron sâu đa tầng (**MLP**):
  $$\phi^{\text{GMF}} = \mathbf{p}_u^G \odot \mathbf{q}_i^G$$
  $$\phi^{\text{MLP}} = a_L \left( W_L^T (\dots a_1(W_1^T [\mathbf{p}_u^M, \mathbf{q}_i^M] + b_1) \dots) + b_L \right)$$
  $$\hat{y}_{ui} = \sigma \left( \mathbf{h}^T [\phi^{\text{GMF}}, \phi^{\text{MLP}}] \right)$$
- **Hàm tối ưu:** Binary Cross-Entropy Loss (BCEWithLogitsLoss) kết hợp tỉ lệ lấy mẫu âm $1:4$ (1 mẫu dương đi kèm 4 mẫu âm ngẫu nhiên).

---

## 3. GIAO THỨC ĐÁNH GIÁ TOP-N KHÁCH QUAN (OFFLINE EVALUATION HARNESS)

Để bảo đảm tính trung thực khoa học tuyệt đối, hệ thống MusicAI sử dụng giao thức đánh giá Top-N chuẩn quốc tế:

### 3.1. Phân hoạch Dữ liệu (Hold-Out Evaluation Protocol)
- Với mỗi người dùng trong tập kiểm thử ($u \in U_{\text{test}}$), tập tương tác được chia thành:
  - $\mathcal{H}_u^{\text{train}}$: Toàn bộ tương tác trong tập huấn luyện (dùng để khớp mô hình).
  - $\mathcal{T}_u^{\text{test}}$: Tập các mục người dùng đã nghe trong tập kiểm thử (Ground Truth Positives).
- Khi tính toán gợi ý Top-$N$, hệ thống loại bỏ triệt để các mục đã có trong $\mathcal{H}_u^{\text{train}}$ (`candidate_items = All_Items \ User_Train_History`).

### 3.2. Định nghĩa Toán học của 5 Độ đo Benchmark

Giả sử $\mathcal{R}_K(u)$ là danh sách $K$ mục có điểm dự đoán cao nhất được mô hình đề xuất cho người dùng $u$:

1. **Hit Rate@K (Tỷ lệ chạm mục tiêu):**
   $$\text{HR@}K = \frac{1}{|U|} \sum_{u \in U} \mathbb{I}\left( |\mathcal{R}_K(u) \cap \mathcal{T}_u^{\text{test}}| > 0 \right)$$
2. **Precision@K (Độ chính xác Top-K):**
   $$\text{Precision@}K = \frac{1}{|U|} \sum_{u \in U} \frac{|\mathcal{R}_K(u) \cap \mathcal{T}_u^{\text{test}}|}{K}$$
3. **Recall@K (Độ bao phủ Top-K):**
   $$\text{Recall@}K = \frac{1}{|U|} \sum_{u \in U} \frac{|\mathcal{R}_K(u) \cap \mathcal{T}_u^{\text{test}}|}{|\mathcal{T}_u^{\text{test}}|}$$
4. **Normalized Discounted Cumulative Gain@K (NDCG@K):**
   Đo lường vị trí xuất hiện của mục phù hợp trong bảng xếp hạng (mục phù hợp càng đứng đầu điểm càng cao):
   $$\text{DCG@}K(u) = \sum_{k=1}^K \frac{2^{\mathbb{I}(r_k \in \mathcal{T}_u^{\text{test}})} - 1}{\log_2(k + 1)}, \quad \text{NDCG@}K(u) = \frac{\text{DCG@}K(u)}{\text{IDCG@}K(u)}$$
5. **Mean Reciprocal Rank@K (MRR@K):**
   Đo lường nghịch đảo vị trí đầu tiên xuất hiện mục trúng đích:
   $$\text{MRR@}K = \frac{1}{|U|} \sum_{u \in U} \frac{1}{\text{rank}_u^{(1)}} \quad (\text{hoặc } 0 \text{ nếu không có Hit})$$

---

## 4. CHIẾN LƯỢC QUẢN LÝ DUNG LƯỢNG VÀ TỐC ĐỘ DỰ ĐOÁN (INFERENCE PERFORMANCE)

| Mô hình | Thời gian huấn luyện dự kiến | Kích thước Checkpoint | Độ trễ suy luận Top-10 |
|---|---|---|---|
| Popularity | < 0.2s | ~50 KB | < 0.1 ms |
| Item-Based CF | ~2.5s | ~8.5 MB | ~1.5 ms |
| User-Based CF | ~3.0s | ~12.0 MB | ~2.0 ms |
| iALS | ~8.0s (15 iterations) | ~1.8 MB | ~0.8 ms (Vector dot product) |
| BPR-MF | ~12.0s (30 epochs SGD) | ~1.8 MB | ~0.8 ms |
| Implicit NeuMF | ~25.0s (20 epochs PyTorch) | ~2.2 MB | ~3.5 ms |

> **Đánh giá tổng quan:** Toàn bộ chuỗi 6 mô hình MusicAI hoàn toàn có thể huấn luyện xong trong vòng dưới 1 phút trên CPU máy tính cá nhân, và kích thước tổng của tất cả checkpoint chỉ khoảng ~26 MB, bảo đảm hệ thống vận hành siêu nhẹ và phản hồi lập tức.
